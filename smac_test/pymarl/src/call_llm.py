import random
import json
import time
import urllib.request

class LLM:
    def __init__(self, mode='openai') -> None:
        if mode == 'openai':
            api_key_list = [
                'sk-KSSWDH0mSBVunRM0FxkkMOgyJjCN21noTP8bZr2lLm1bPP7V',
                'sk-DVbHrrZAIqbcRZIRUGm8Yayw9jcU7l0scT5JlcEctx4ElfJz' # fill the key
            ]
            self.agent_big = gpt_agent(random.choice(api_key_list), api_key_list, model_name='gpt-4o-2024-08-06')
            self.agent_small = gpt_agent(random.choice(api_key_list), api_key_list, model_name='gpt-4o-mini')
            self.call_llm = self.call_llm_openai
        else:
            assert 0

    def call_llm_openai(self, prompt, big_model=False, temperature=0.0):
        if big_model:
            return self.agent_big.ask(prompt)
        else:
            return self.agent_small.ask(prompt)

class gpt_agent():

    def __init__(self, api_key:str, api_key_list, model_name="gpt-3.5-turbo"):
        self.base_url = "https://api.chatanywhere.tech/v1"
        self.api_key = api_key
        self.ask_call_cnt = 0
        self.ask_call_cnt_sup = 3
        self.model_name = model_name
        self.api_key_list = api_key_list

    def ask(self, question, temperature=0.0, stop=None) -> str:
        res = "No answer!"
        self.ask_call_cnt = self.ask_call_cnt + 1
        if self.ask_call_cnt > self.ask_call_cnt_sup:
            print("======> Achieve call count limit, Return!")
            self._random_key()
            self.ask_call_cnt = 0
            return res

        messages = [{"role": "user", "content": question}]
        payload = {"model": self.model_name, "messages": messages}
        if stop is not None:
            payload["stop"] = stop

        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
            )
            resp = urllib.request.urlopen(req, timeout=60)
            rsp = json.loads(resp.read().decode('utf-8'))
            res = rsp["choices"][0]["message"]["content"]
            self.ask_call_cnt = 0
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8', errors='ignore')
            if e.code == 401:
                self._random_key()
                print(f"======> AuthenticationError (401): {body}")
            elif e.code == 429:
                print(f"======> {self.api_key} <===== \nAchieve ChatGPT rate limit, sleep!")
                self._random_key()
                time.sleep(10)
                return self.ask(question)
            elif e.code >= 500:
                print(f"======> Server error ({e.code}), will retry after 10 seconds")
                self._random_key()
                time.sleep(10)
                return self.ask(question)
            else:
                print(f"======> HTTP {e.code}: {body}")
                self._random_key()
        except Exception as e:
            print(f"======> Exception: {e}")
            self._random_key()
            if "HTTPSConnectionPool" in str(e):
                time.sleep(60)
        return res

    def _random_key(self) -> None:
        self.api_key = random.choice(self.api_key_list)
