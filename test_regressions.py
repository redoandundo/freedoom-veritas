"""Offline regression checks. Never call Gemini or mutate live project state."""
import asyncio
import copy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import server as app

class Request:
    def __init__(self, body): self.body = body
    async def json(self): return self.body

class EvaluationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.rules = copy.deepcopy(app.DEFAULT_CHECKLIST)
        self.criteria = [{'criterion_id':r['id'], 'value':False if r['kind']=='flag' else (True if type(r['allowed_values'][0]) is bool else max(v for v in r['allowed_values'] if v is not None))} for r in self.rules]

    def test_weighted_na(self):
        for c in self.criteria:
            if c['criterion_id']=='dialogue': c['value']=7
        self.assertEqual(app.calculate_score(self.criteria,self.rules)['final_score'],95.5)
        for c in self.criteria:
            if c['criterion_id'] in ['discovery','identification']: c['value']=None
            if c['criterion_id']=='closure': c['value']=False
            if c['criterion_id']=='dialogue': c['value']=10
        self.assertEqual(app.calculate_score(self.criteria,self.rules)['final_score'],92.31)

    def test_evidence_guards_and_custom_rules(self):
        record={"language":"RU","transcript":[
            {"speaker":"operator","text":"Фридом брокер [Имя скрыто], здравствуйте!"},
            {"speaker":"client","text":"Спасибо"},
            {"speaker":"system","text":"Клиент завершил разговор."},
            {"speaker":"operator","text":"Подскажите, у вас ещё вопросы?"}]}
        criteria=[{"criterion_id":"contact","value":False,"rationale":"Нет имени","evidence_turns":[1]},
                  {"criterion_id":"closure","value":False,"rationale":"Нет прощания","evidence_turns":[4]}]
        changes=app.apply_evidence_guards(criteria,record,self.rules)
        self.assertTrue(criteria[0]["value"])
        self.assertIsNone(criteria[1]["value"])
        self.assertEqual(len(changes),2)
        self.assertFalse(changes[0]["original"]["value"])
        custom=copy.deepcopy(self.rules)
        for r in custom: r["rule_text"]="Custom rule"
        original=copy.deepcopy(criteria);original[0]["value"]=False;original[1]["value"]=False
        self.assertEqual(app.apply_evidence_guards(original,record,custom),[])
        record["transcript"][0]["text"]="Здравствуйте! Ваше имя [Имя скрыто]? До свидания!"
        self.assertEqual(app.apply_evidence_guards(original,record,self.rules),[])

    def test_navigation_is_not_clarification(self):
        record={"language":"KZ","transcript":[
            {"speaker":"client","text":"Деректемелер қайда?"},
            {"speaker":"operator","text":"Меню ашып, валютаны таңдап, деректемелерді жүктеп алыңыз."}]}
        criterion={"criterion_id":"discovery","value":True,"rationale":"Clarified","evidence_turns":[2]}
        app.apply_evidence_guards([criterion],record,self.rules)
        self.assertIsNone(criterion["value"])
        criterion["value"]=False
        record["transcript"][1]["text"]="Қай валюта? Меню ашыңыз."
        self.assertEqual(app.apply_evidence_guards([criterion],record,self.rules),[])

    async def test_appeal_preserves_original_and_recalculates(self):
        for c in self.criteria:
            c.update(rationale="Synthetic test",evidence_turns=[1])
        result={"conversation_id":"synthetic","status":"ok","criteria":self.criteria,"score":app.calculate_score(self.criteria,self.rules)}
        state={"results":{"synthetic":result},"appeals":[],"run_checklist":self.rules}
        with patch.object(app,"state",state),patch.object(app,"progress",{"running":False}),patch.object(app,"persist"):
            await app.appeal(Request({"conversation_id":"synthetic","criterion_id":"closure","comment":"Synthetic appeal"}))
            await app.resolve_appeal(Request({"conversation_id":"synthetic","criterion_id":"closure","new_value":False,"reason":"Synthetic decision"}))
            self.assertEqual(result["score"]["final_score"],95)
            self.assertEqual(result["original_ai"]["score"]["final_score"],100)
            self.assertEqual(state["appeals"][0]["status"],"reviewed")
            exported=app.export_result(result)
            self.assertNotIn("original_ai",exported)
            self.assertEqual(exported["status"],"ok")

    def test_partial_run_preserves_last_complete_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); target=root/"results/dev/results.json"
            target.parent.mkdir(parents=True);target.write_text("previous")
            state={"group":"DEV","records":[{"conversation_id":"a"}],"results":{},"model":app.MODEL}
            with patch.object(app,"state",state),patch.object(app,"ROOT",root),patch.object(app,"DATA",root/"data"):
                app.persist();self.assertEqual(target.read_text(),"previous")
                state["results"]["a"]={"conversation_id":"a","status":"ok","criteria":self.criteria,"score":app.calculate_score(self.criteria,self.rules),"evaluator_version":app.EVALUATOR_VERSION}
                app.persist();self.assertIn('"team_name": "Veritas"',target.read_text())

    async def test_daily_quota_rejects_before_mutation(self):
        state={"records":[{"conversation_id":"a"}],"results":{"a":{"status":"ok"}}}
        with patch.object(app,"state",state),patch.object(app,"progress",{"running":False}),patch.object(app,"verification",{"running":False}),patch.object(app.limiter,"daily_until",app.time.monotonic()+1000):
            with self.assertRaises(app.HTTPException) as ctx:
                await app.evaluate(Request({"api_key":"offline-test"}),None)
            self.assertEqual(ctx.exception.status_code,429)
            self.assertEqual(state["results"]["a"]["status"],"ok")
        response=app.httpx.Response(429,json={"error":{"details":[{"violations":[{"quotaId":"RequestsPerDay"}]}]}})
        self.assertTrue(app.is_daily_quota(response))
        response=app.httpx.Response(429,json={"error":{"details":[{"violations":[{"quotaId":"RequestsPerMinute"}]}]}})
        self.assertFalse(app.is_daily_quota(response))

    async def test_daily_quota_cancels_waiting_workers(self):
        state={"results":{},"appeals":[],"records":[{"conversation_id":"a"},{"conversation_id":"b"}]}
        async def fake(client,record,*args):
            if record["conversation_id"]=="a": raise app.DailyQuotaExceeded("Daily quota exhausted")
            await asyncio.Event().wait()
        with patch.object(app,"state",state),patch.object(app,"progress",{"running":True}),patch.object(app,"evaluate_one",fake),patch.object(app,"persist"),patch.object(app,"save"):
            await asyncio.wait_for(app.evaluate_background("offline",self.rules,state["records"]),1)
            self.assertFalse(app.progress["running"])
            self.assertTrue(app.progress["cancelled"])
            self.assertIn("Daily quota",app.progress["error"])

    async def test_selected_preserves_peer_and_archives(self):
        state={'records':[{'conversation_id':'a'},{'conversation_id':'b'}], 'results':{'a':{'status':'ok','marker':'old'},'b':{'status':'ok','marker':'peer'}},'appeals':[],'run_checklist':self.rules,'model':app.MODEL,'group':'DEV','archives':[]}
        async def fake(*args):
            return {'conversation_id':'a','status':'ok','criteria':self.criteria,'score':app.calculate_score(self.criteria,self.rules),'evaluator_version':app.EVALUATOR_VERSION}
        with patch.object(app,'state',state),patch.object(app,'checklist',self.rules),patch.object(app,'progress',{'running':False}),patch.object(app,'verification',{'running':False}),patch.object(app,'save'),patch.object(app,'persist'),patch.object(app,'evaluate_one',fake),patch.object(app.limiter,'daily_until',0):
            await app.evaluate(Request({'api_key':'offline-test','conversation_id':'a'}),None)
            await app.active_evaluation
            self.assertEqual(state['results']['b']['marker'],'peer')
            self.assertEqual(state['single_history'][0]['result']['marker'],'old')
            self.assertEqual(app.progress['done'],2)
            self.assertEqual(app.progress['completed_this_run'],1)
            self.assertEqual(len(app.export_payload()['results']),2)
            with self.assertRaises(app.HTTPException):
                await app.evaluate(Request({'api_key':'offline-test','conversation_id':'missing'}),None)

if __name__=='__main__': unittest.main()
