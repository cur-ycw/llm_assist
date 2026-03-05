import json
import textwrap
import random
import os
import subprocess
import time
from tensorboard.backend.event_processing import event_accumulator
import shutil
import numpy as np
import datetime

from call_llm import LLM

class SemDiv():
    def __init__(self, env, env_file, timing):
        # 获取项目根目录路径（基于当前文件位置）
        _THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../SemDiv/language
        self.root_dir = os.path.dirname(_THIS_DIR)              # .../SemDiv
        
        self.llm = LLM(mode='openai')
        self.n_tm_total = 6
        self.cuda_id = 0
        self.name = f'semdiv'

        self.big_model = True
        self.few_shot = True
        self.start_few_shot_round = 0
        self.max_attempt_behavior = 2
        self.max_attempt_behavior_total = 30
        self.involve_ego = True
        self.do_traj_check = True
        self.do_similarity_check = True
        self.load_agent_id = 1 # only for football
        
        self.timing = timing
        self.behavior_library = {}
        self.tm_library = {}
        self.ego_library = {}
        self.env = env
        self.env_file = env_file
        self.env_idx = 1
        self.lib_dir = f'lib/{self.timing}_{self.name}'
        os.makedirs(self.lib_dir, exist_ok=True)
        os.system(f'cp {os.path.basename(__file__)} {self.lib_dir}/rhs.py')

        if 'lbf' in self.env:
            self.check_status = self.check_status_pymarl
            self.tm_performance_tolerance = 0.3
            self.ego_performance_tolerance = 0.2
            self.ego_performance_additional_tolerance = 0.2
        elif 'pp' in self.env:
            self.check_status = self.check_status_pymarl
            self.tm_performance_tolerance = 0.6
            self.ego_performance_tolerance = 0.2
            self.ego_performance_additional_tolerance = 0.2
        elif 'sc2' in self.env:
            self.check_status = self.check_status_pymarl
            self.tm_performance_tolerance = 0.5
            self.ego_performance_tolerance = 0.2
            self.ego_performance_additional_tolerance = 0.2
        elif 'football' in self.env:
            self.check_status = self.check_status_football
            self.tm_performance_tolerance = 0.5
            self.ego_performance_tolerance = 0.2
            self.ego_performance_additional_tolerance = 0.2
        else:
            assert 0

    def run(self):
        self.tm_idx = 1
        behavior_idx = 0
        while True:
            behavior_idx += 1
            self.behavior_idx = behavior_idx
            while True:
                behavior_prompt, behavior_output = self.generate_behavior()
                try:
                    assert behavior_output != 'No answer!'
                    behavior = self.llm.call_llm(f'Task: Find and extract the part describing the specific cooperation preference from the original text, ignoring other analysis related content. Output the cooperation preference only.\n\nInput: ({behavior_output})\n\nOutput: ', big_model=False)
                    break
                except:
                    pass
            self.behavior_library[behavior_idx] = {}
            self.behavior_library[behavior_idx]['prompt'] = behavior_prompt
            self.behavior_library[behavior_idx]['behavior_output'] = behavior_output
            self.behavior_library[behavior_idx]['behavior'] = behavior
            self.behavior_library[behavior_idx]['attempt_history'] = {}
            self.tm_library[self.tm_idx] = {}
            self.ego_library[self.tm_idx] = {}
            self.store_log_files()
            n_attempt_behavior = 1
            self.n_attempt_behavior = n_attempt_behavior
            while True:
                print(self.tm_idx, behavior_idx, n_attempt_behavior) #(队友_idx, 队友_行为_idx, 队友_行为_尝试_idx)
                if behavior_idx >= self.max_attempt_behavior_total: # 队友_行为_idx 超过最大尝试次数
                    print('what can i say')
                    return
                code_prompt = self.generate_code_prompt(behavior, n_attempt_behavior, behavior_idx)
                code_llm_output, code = self.llm_write_code(code_prompt, behavior)
                self.behavior_library[behavior_idx]['attempt_history'][n_attempt_behavior] = {}
                self.behavior_library[behavior_idx]['attempt_history'][n_attempt_behavior]['time'] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                self.behavior_library[behavior_idx]['attempt_history'][n_attempt_behavior]['prompt'] = code_prompt
                self.behavior_library[behavior_idx]['attempt_history'][n_attempt_behavior]['output'] = code_llm_output
                self.behavior_library[behavior_idx]['attempt_history'][n_attempt_behavior]['code'] = code
                self.store_log_files()
                self.traj_check_prompt = ''
                self.traj_check_output = ''
                self.traj_check_info = ''
                
                # Verify that the llm function exists in environment.py before training
                # This is critical for tm2, tm3, etc. to ensure the code is available when training starts
                max_verify_attempts = 10
                verify_interval = 0.5  # seconds
                llm_func_found = False
                for attempt in range(max_verify_attempts):
                    with open(self.env_file, 'r') as f:
                        file_content = f.read()
                    if f'def llm{self.tm_idx}(self' in file_content:
                        llm_func_found = True
                        print(f"[DEBUG] Verified: llm{self.tm_idx} function exists in environment.py (attempt {attempt+1})")
                        break
                    else:
                        print(f"[WARNING] llm{self.tm_idx} function not found, waiting... (attempt {attempt+1}/{max_verify_attempts})")
                        time.sleep(verify_interval)
                
                if not llm_func_found:
                    # Mark this attempt as failed and continue to next attempt
                    self.behavior_library[behavior_idx]['attempt_history'][n_attempt_behavior]['status'] = 'bug'
                    self.behavior_library[behavior_idx]['attempt_history'][n_attempt_behavior]['info'] = f'llm{self.tm_idx} function not found in environment.py after {max_verify_attempts} attempts. Code may not have been written correctly.'
                    self.store_log_files()
                    # Restore the env file and try next attempt
                    with open(self.env_file, 'w') as f:
                        f.write(self.original_env_file)
                    n_attempt_behavior += 1
                    self.n_attempt_behavior = n_attempt_behavior
                    if n_attempt_behavior > self.max_attempt_behavior:
                        self.behavior_library[behavior_idx]['status'] = 'failed'
                        self.store_log_files()
                        break
                    continue
                
                # Additional small delay to ensure file system is fully synced before training starts
                time.sleep(0.5)
                
                cmd = self.train_tm()
                status, info, model_path = self.check_status()
                self.behavior_library[behavior_idx]['attempt_history'][n_attempt_behavior]['cmd'] = cmd
                self.behavior_library[behavior_idx]['attempt_history'][n_attempt_behavior]['status'] = status # bug, fail, constant, similar, success
                self.behavior_library[behavior_idx]['attempt_history'][n_attempt_behavior]['info'] = info
                self.behavior_library[behavior_idx]['attempt_history'][n_attempt_behavior]['traj_check_prompt'] = self.traj_check_prompt
                self.behavior_library[behavior_idx]['attempt_history'][n_attempt_behavior]['traj_check_output'] = self.traj_check_output
                self.store_log_files()
                if status == 'success':
                    self.behavior_library[behavior_idx]['status'] = 'success'
                    self.behavior_library[behavior_idx]['traj_check_info'] = self.traj_check_info
                    self.tm_library[self.tm_idx]['behavior'] = behavior
                    self.tm_library[self.tm_idx]['code'] = code
                    self.tm_library[self.tm_idx]['model_path'] = model_path
                    self.store_log_files()
                    if self.involve_ego:
                        cmd, model_path = self.train_ego()
                        self.ego_library[self.tm_idx]['cmd'] = cmd
                        self.ego_library[self.tm_idx]['model_path'] = model_path
                        self.ego_library[self.tm_idx]['behavior'] = behavior
                    self.store_log_files()
                    self.tm_idx += 1
                    break

                # restore the env file
                with open(self.env_file, 'w') as f:
                    f.write(self.original_env_file)
                n_attempt_behavior += 1
                self.n_attempt_behavior = n_attempt_behavior
                if n_attempt_behavior > self.max_attempt_behavior:
                    if status == 'similar':
                        self.behavior_library[behavior_idx]['status'] = 'similar'
                    else:
                        self.behavior_library[behavior_idx]['status'] = 'failed'
                    self.store_log_files()
                    break
            if self.tm_idx > self.n_tm_total:
                print('Mission accomplished.')
                break

    def generate_behavior(self):
        prompt = prompt_basic_env + prompt_code + prompt_multi_modality
        if (not self.few_shot) or (self.tm_idx < self.start_few_shot_round) or (self.tm_idx == 1):
            prompt += prompt_behavior_first
        elif self.few_shot and self.tm_idx >= 2:

            behavior_indexes = self.extract_examples(type='success')
            if len(behavior_indexes) > 0:
                prompt += "Here are some behavior examples:\n"
                for i, behavior_idx in enumerate(behavior_indexes):
                    prompt += f" - Example {i+1}: [{self.behavior_library[behavior_idx]['behavior']}\nActual running behaviors: {self.behavior_library[behavior_idx]['traj_check_info']}]\n"
                    
            behavior_indexes = self.extract_examples(type='similar')
            if len(behavior_indexes) > 0:
                prompt += "Here are some behavior examples that are feasible but not novel:\n"
                for i, behavior_idx in enumerate(behavior_indexes):
                    behavior = self.behavior_library[behavior_idx]['behavior']
                    ego2info = self.behavior_library[behavior_idx]['attempt_history'][self.max_attempt_behavior]['info']
                    prompt += f" - Not-novel example {i+1}: [{behavior}\nThis behavior is the same as:\n"
                    similar_count = 1
                    for ego_info in ego2info.values():
                        if type(ego_info) == type({}) and ego_info['is_similar']:
                            similar_behavior = ego_info['behavior']
                            prompt += f"    {similar_count}: {similar_behavior}\n"
                            similar_count += 1
                    prompt += '\n]\n'
                            
            prompt += prompt_behavior
            
        output = self.llm.call_llm(prompt, big_model=self.big_model)
        
        return prompt, output
    
    def extract_examples(self, type):
        behavior_indexes = []
        for behavior_idx, behavior in self.behavior_library.items():
            if 'status' in behavior and behavior['status'] == type:
                behavior_indexes.append(behavior_idx)
        return behavior_indexes
    
    def generate_code_prompt(self, behavior, n_attempt_behavior, behavior_idx):
        prompt = prompt_basic_env + prompt_code + f"\nNow we want to train a team with this specific cooperation behavior:\n---\n{behavior}\n---" + prompt_write_code
        if n_attempt_behavior > 1:
            prompt += "\nWe have tried some reward function code before, but they are not good enough:\n"
            for previous_attempt in range(1, n_attempt_behavior):
                attempt_code = self.behavior_library[behavior_idx]['attempt_history'][previous_attempt]['code']
                attempt_status = self.behavior_library[behavior_idx]['attempt_history'][previous_attempt]['status']
                attempt_info = self.behavior_library[behavior_idx]['attempt_history'][previous_attempt]['info']
                prompt += f"Attempt {previous_attempt}: [\n{attempt_code}\n]\n"
                if attempt_status == 'bug':
                    prompt += f"This code has some bugs with these messages: ...\n{attempt_info}\n"
                elif attempt_status == 'failed':
                    prompt += f"The team trained with this additional reward function failed to complete the original task. Original return: {attempt_info}\n"
                elif attempt_status == 'constant':
                    if 'lbf' in self.env or 'pp' in self.env or 'football' in self.env:
                        prompt += f"The values of this additional reward function are consistently close to {attempt_info}, meaning that the team is not able to optimize it as it is written.\n"
                elif attempt_status == 'misaligned':
                    reason = self.llm.call_llm(f"Task: Summarize the reasons why this expert judged misalignment, in less than 30 words.\n\nInput:\n{attempt_info[1]}\n\nOutput: ", big_model=False)
                    prompt += f"An expert claimed that, the actual behavior of the team trained by this reward function code, does not align with the cooperation preference. Reason: [{reason}]\n"
                elif attempt_status == 'similar':
                    prompt += f"The team trained with this reward function is too similar with:\n"
                    similar_count = 1
                    for ego_info in attempt_info.values():
                        if type(ego_info) == type({}) and ego_info['is_similar']:
                            similar_behavior = ego_info['behavior']
                            prompt += f" - Team {similar_count}: {similar_behavior}\n"
                            similar_count += 1
                else:
                    print(attempt_status)
                    assert 0
            prompt += "\nBased on these information, You may consider change the function, like changing the reward components or their scales."
        return prompt

    def llm_write_code(self, code_prompt, behavior):
        while True:
            code_llm_output = self.llm.call_llm(code_prompt, big_model=self.big_model)
            try:
                if "def additional_reward(self" not in code_llm_output:
                    continue
                code = self.llm.call_llm(code_llm_output + '\n\nTODO: Extract the python function `additional_reward` in the text. Output the function only (start with "```python\ndef", end with "\n```") so that i can directly copy it into my code.', big_model=False)
                store_code = code.split("```python\n")[1].split('```')[0].replace("def additional_reward(self", f"def llm{self.tm_idx}(self")
                if 'pp' in self.env:
                    store_code = store_code.replace(f'def llm{self.tm_idx}(self)', f'def llm{self.tm_idx}(self, agent, world)')
                if store_code is None:
                    continue
                timing = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                write_code = f"\n'''\n{timing}\n{behavior}\n'''\n{store_code}\n"
                with open(self.env_file, 'r') as f:
                    self.original_env_file = f.read()
                with open(self.env_file, "a") as f:
                    f.write(textwrap.indent(write_code, '    '))
                    # Force file system sync to ensure the code is written to disk
                    try:
                        f.flush()
                        os.fsync(f.fileno())
                    except:
                        pass
                # Verify the code was written correctly
                print(f"[DEBUG] Written llm{self.tm_idx} function to {self.env_file} at {timing}")
                # Verify the function exists in the file
                with open(self.env_file, 'r') as f:
                    file_content = f.read()
                    if f'def llm{self.tm_idx}(self' in file_content:
                        print(f"[DEBUG] Verified: llm{self.tm_idx} function exists in environment.py")
                    else:
                        print(f"[WARNING] llm{self.tm_idx} function not found in environment.py after writing!")
                break
            except:
                pass
        return code_llm_output, store_code
    
    def train_tm(self):
        if 'lbf' in self.env:
            self.run_name = f'{self.name}_tm_{self.tm_idx}'
            t_max = 50000
            script_path = os.path.join(self.root_dir, 'pymarl/src/scripts/semdiv_train_tm_sp_llm.sh')
            cmd = f'bash {script_path} {self.cuda_id} {self.tm_idx} {self.run_name} {t_max} {self.env_idx} {self.env}'
            print(cmd)
            os.system(cmd)
        elif 'pp' in self.env:
            self.run_name = f'{self.name}_tm_{self.tm_idx}'
            t_max = 500000
            script_path = os.path.join(self.root_dir, 'pymarl/src/scripts/semdiv_train_tm_sp_llm.sh')
            cmd = f'bash {script_path} {self.cuda_id} {self.tm_idx} {self.run_name} {t_max} {self.env_idx} {self.env}'
            print(cmd)
            os.system(cmd)
        elif 'sc2' in self.env:
            self.run_name = f'{self.name}_tm_{self.tm_idx}'
            t_max = 1000000
            script_path = os.path.join(self.root_dir, 'pymarl/src/scripts/semdiv_train_tm_sp_llm.sh')
            cmd = f'bash {script_path} {self.cuda_id} {self.tm_idx} {self.run_name} {t_max} {self.env_idx} {self.env}'
            print(cmd)
            os.system(cmd)
        elif 'football' in self.env:
            self.run_name = f'{self.name}_tm_{self.tm_idx}'
            t_max = 10000000
            self.map = self.env.split(':')[-1]
            script_path = os.path.join(self.root_dir, 'HARL/examples/train_semdiv_tm.sh')
            cmd = f'bash {script_path} {self.cuda_id} {self.tm_idx} {self.run_name} {t_max} {self.env_idx} {self.map} {np.random.randint(100)}'
            print(cmd)
            os.system(cmd)
        else:
            assert 0
        while True:
            time.sleep(300)
            print(f'check running [tm {self.tm_idx, self.behavior_idx, self.n_attempt_behavior}]: {datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
            if not self.is_process_running():
                break
        return cmd

    def train_ego(self):
        if 'lbf' in self.env:
            self.ego_run_name = f'{self.name}_multihead_{self.tm_idx}'
            t_max = 300000
            tm_model_paths = ','.join([self.tm_library[i]['model_path'].split('pymarl/')[1] for i in range(1, self.tm_idx+1)]) # loaded all tm models, but only train with the latest one in pymarl codes.
            last_ego_model_path = self.ego_library[self.tm_idx-1]['model_path'].split('pymarl/')[1] if self.tm_idx > 1 else ""
            reg_coef = 500
            script_path = os.path.join(self.root_dir, 'pymarl/src/scripts/semdiv_train_ego.sh')
            cmd = f'bash {script_path} {self.cuda_id} {self.ego_run_name} {self.tm_idx} {t_max} {reg_coef} {self.env_idx} {self.env} {tm_model_paths} {last_ego_model_path}'
            print(cmd)
            os.system(cmd)
            while True:
                time.sleep(300)
                print(f'check running [ego {self.tm_idx}]: {datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
                if not self.is_process_running(mode='ego'):
                    break
                
        elif 'pp' in self.env:
            self.ego_run_name = f'{self.name}_multihead_{self.tm_idx}'
            t_max = 500000
            tm_model_paths = ','.join([self.tm_library[i]['model_path'].split('pymarl/')[1] for i in range(1, self.tm_idx+1)]) # loaded all tm models, but only train with the latest one in pymarl codes.
            last_ego_model_path = self.ego_library[self.tm_idx-1]['model_path'].split('pymarl/')[1] if self.tm_idx > 1 else ""
            reg_coef = 500
            script_path = os.path.join(self.root_dir, 'pymarl/src/scripts/semdiv_train_ego.sh')
            cmd = f'bash {script_path} {self.cuda_id} {self.ego_run_name} {self.tm_idx} {t_max} {reg_coef} {self.env_idx} {self.env} {tm_model_paths} {last_ego_model_path}'
            print(cmd)
            os.system(cmd)
            while True:
                time.sleep(300)
                print(f'check running [ego {self.tm_idx}]: {datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
                if not self.is_process_running(mode='ego'):
                    break
                
        elif 'sc2' in self.env:
            self.ego_run_name = f'{self.name}_multihead_{self.tm_idx}'
            t_max = 500000
            tm_model_paths = ','.join([self.tm_library[i]['model_path'].split('pymarl/')[1] for i in range(1, self.tm_idx+1)]) # loaded all tm models, but only train with the latest one in pymarl codes.
            last_ego_model_path = self.ego_library[self.tm_idx-1]['model_path'].split('pymarl/')[1] if self.tm_idx > 1 else ""
            reg_coef = 500
            script_path = os.path.join(self.root_dir, 'pymarl/src/scripts/semdiv_train_ego.sh')
            cmd = f'bash {script_path} {self.cuda_id} {self.ego_run_name} {self.tm_idx} {t_max} {reg_coef} {self.env_idx} {self.env} {tm_model_paths} {last_ego_model_path}'
            print(cmd)
            os.system(cmd)
            while True:
                time.sleep(300)
                print(f'check running [ego {self.tm_idx}]: {datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
                if not self.is_process_running(mode='ego'):
                    break
                
        elif 'football' in self.env:
            self.ego_run_name = f'{self.name}_multihead_{self.tm_idx}'
            t_max = 10000000
            # tm_model_paths = ','.join([self.tm_library[i]['model_path'].split('HARL/examples/')[1] for i in range(1, self.tm_idx+1)]) # all seen ones
            tm_model_paths = self.tm_library[self.tm_idx]['model_path'].split('HARL/examples/')[1]
            last_ego_model_path = self.ego_library[self.tm_idx-1]['model_path'].split('HARL/examples/')[1] if self.tm_idx > 1 else ""
            use_reg = len(last_ego_model_path) > 0
            # reg_coef = 500
            script_path = os.path.join(self.root_dir, 'HARL/examples/train_semdiv_ego.sh')
            cmd = f'bash {script_path} {self.cuda_id} {self.tm_idx} {self.ego_run_name} {t_max} {self.env_idx} {self.map} {use_reg} {self.load_agent_id} {tm_model_paths} {last_ego_model_path}'
            print(cmd)
            os.system(cmd)
            while True:
                time.sleep(180)
                print(f'check running [ego {self.tm_idx}]: {datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
                if not self.is_process_running(mode='ego'):
                    break
                
        else:
            assert 0

        if 'lbf' in self.env or 'pp' in self.env or 'sc2' in self.env:
            if 'lbf' in self.env or 'pp' in self.env:
                path = os.path.join(self.root_dir, 'pymarl/results/pymarl/gymma', self.ego_run_name)
            elif 'sc2' in self.env:
                path = os.path.join(self.root_dir, 'pymarl/results/pymarl/sc2_v2', self.ego_run_name)
            else:
                assert 0
            path = os.path.join(path, os.listdir(path)[0])
            sacred_path = os.path.join(path, 'sacred', '1')
            config_path = os.path.join(sacred_path, 'config.json')
            f = open(config_path)
            config = json.load(f)
            f.close()
            seed = config['seed']
            model_path = None
            model_path_ = os.path.join(self.root_dir, 'pymarl/results/models')
            for model in os.listdir(model_path_):
                if model.startswith(f'seed_{str(seed)}_{self.ego_run_name}'):
                    model_path = os.path.join(model_path_, model)
                    break
        elif 'football' in self.env:
            model_path = os.path.join(self.root_dir, f'HARL/examples/results/football/{self.map}/mappo/{self.ego_run_name}')
            assert len(os.listdir(model_path)) == 2, os.listdir(model_path)
            sub_path = [x for x in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, x))]
            assert len(sub_path) == 1, sub_path
            model_path = os.path.join(model_path, sub_path[0], 'models')
        else:
            assert 0

        assert model_path is not None
        return cmd, model_path

    def store_log_files(self):
        os.system(f'cp {self.env_file} {self.lib_dir}/env.py')
        with open(f'{self.lib_dir}/behavior.json', 'w') as file:
            json.dump(self.behavior_library, file)
        with open(f'{self.lib_dir}/teammate.json', 'w') as file:
            json.dump(self.tm_library, file)
        if self.involve_ego:
            with open(f'{self.lib_dir}/ego.json', 'w') as file:
                json.dump(self.ego_library, file)

    def is_process_running(self, mode=''):
        if 'lbf' in self.env or 'pp' in self.env or 'sc2' in self.env:
            command = f'ps -ef | grep -i {self.run_name}'
            if mode == 'ego':
                command = f'ps -ef | grep -i {self.ego_run_name}'
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                outputs = result.stdout
                process_string = f'--name={self.run_name}'
                if mode == 'ego':
                    process_string = f'--name={self.ego_run_name}'
                return process_string in outputs
            else:
                print(result.stderr)
                assert 0
        if 'football' in self.env:
            command = f'ps -ef | grep -i {self.run_name}'
            if mode == 'ego':
                command = f'ps -ef | grep -i {self.ego_run_name}'
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                outputs = result.stdout
                process_string = f'mappo-football-{self.run_name}'
                if mode == 'ego':
                    process_string = f'mappo-football-{self.ego_run_name}'
                if process_string not in outputs:
                    return False
            else:
                print(result.stderr)
                assert 0
            path = os.path.join(self.root_dir, f'HARL/examples/results/football/{self.map}/mappo/{self.run_name}/')
            assert os.path.exists(path), path
            out_path = os.path.join(path, '1.out')
            if not os.path.exists(out_path):
                return False
            with open(out_path, 'r') as file:
                message = file.read()
            if 'Traceback (most recent call last):' in message or 'Error: ' in message:
                return False
            assert len(os.listdir(path)) == 2, (path, os.listdir(path))
            sub_path = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
            assert len(sub_path) == 1, sub_path
            model_path = os.path.join(path, sub_path[0], 'models')
            models_str = '-'.join(os.listdir(model_path))
            if hasattr(self, 'last_models_str'):
                if models_str == self.last_models_str and time.time() - self.last_models_str_time > 60 * 10: # no changes after 10 min
                    return False
            self.last_models_str = models_str
            self.last_models_str_time = time.time()
            return True
        else:
            assert 0

    def check_status_pymarl(self):
        if 'lbf' in self.env or 'pp' in self.env:
            pymarl_env = 'gymma'
        elif 'sc2' in self.env:
            pymarl_env = 'sc2_v2'
        else:
            assert 0
        # 训练结果目录：pymarl/results/pymarl/{pymarl_env}/{run_name}/seed_.../
        path = os.path.join(self.root_dir, f'pymarl/results/pymarl/{pymarl_env}', self.run_name)
        if (not os.path.exists(path)) or len(os.listdir(path)) == 0:
            # 训练结果目录还没生成或训练一开始就失败
            return 'bug', f'result dir not found: {os.path.abspath(path)}', None
        path = os.path.join(path, sorted(os.listdir(path))[-1])
        sacred_path = os.path.join(path, 'sacred', '1')
        run_file_path = os.path.join(sacred_path, 'run.json')
        config_path = os.path.join(sacred_path, 'config.json')
        tb_path = os.path.join(path, 'tb_logs')
        if (not os.path.exists(tb_path)) or len(os.listdir(tb_path)) == 0:
            return 'bug', f'tensorboard logs not found: {os.path.abspath(tb_path)}', None
        tb_path = os.path.join(tb_path, sorted(os.listdir(tb_path))[0])
        f = open(config_path)
        config = json.load(f)
        f.close()
        seed = config['seed']
        model_path_ = os.path.join(self.root_dir, 'pymarl/results/models')
        model_path = None
        if os.path.exists(model_path_):
            for model in os.listdir(model_path_):
                if model.startswith(f'seed_{str(seed)}_{self.run_name}'):
                    model_path = os.path.join(model_path_, model)
                    break

        def clean_up(_model_path):
            shutil.rmtree(os.path.join(self.root_dir, f'pymarl/results/pymarl/{pymarl_env}', self.run_name))
            if _model_path is not None and os.path.exists(_model_path):
                shutil.rmtree(_model_path)

        if (not os.path.exists(run_file_path)) or (not os.path.exists(config_path)):
            return 'bug', f'sacred logs not complete under {os.path.abspath(sacred_path)}', None

        f = open(run_file_path)
        data = json.load(f)
        f.close()
        if data['status'] == 'FAILED':
            clean_up(model_path)
            return 'bug', '\n'.join(data['fail_trace'][-3:]), None

        ea = event_accumulator.EventAccumulator(tb_path)
        ea.Reload()
        
        scalar_data = ea.scalars.Items('test_return_original_mean')
        try:
            values = [scalar_data[i][2] for i in range(len(scalar_data))]
        except:
            values = [scalar_data[i].value for i in range(len(scalar_data))]
        performance = np.mean(values[-5:])
        if 'lbf' in self.env or 'pp' in self.env:
            prior_performance = 1.0
        elif 'sc2' in self.env:
            prior_performance = 10.0
        else:
            assert 0
        self.baseline = prior_performance
        baseline = self.baseline
        if performance < baseline * (1 - self.tm_performance_tolerance):
            clean_up(model_path)
            return 'failed', str(performance), None
        
        scalar_data = ea.scalars.Items('test_return_additional_mean')
        try:
            values = [scalar_data[i][2] for i in range(len(scalar_data))]
        except:
            values = [scalar_data[i].value for i in range(len(scalar_data))]
        performance_additional = np.mean(values[-5:])
        performance_additional_init = np.mean(values[:5])
        if (performance_additional - performance_additional_init) / (abs(performance_additional_init) + 1e-5) < 0.1: # no improvement
            clean_up(model_path)
            return 'constant', str(performance_additional_init), None
        
        if self.do_traj_check:
            n_traj = 3
            behavior = self.behavior_library[self.behavior_idx]['behavior']
            input_check_prompt = prompt_basic_env + prompt_code + f"\nWe tried to train a team with this specific cooperation behavior:\n---\n{behavior}\n---\n"
            input_check_prompt += f"After training the team with this reward function, we ran it for {n_traj} episodes: [\n"
            tm_model_path = model_path.split('pymarl/')[1]
            name = f'eval_{self.run_name}_tm_{self.tm_idx}_behavior_{self.behavior_idx}_attempt_{self.n_attempt_behavior}_sp'
            
            # 运行评估脚本并检查返回码
            script_path = os.path.join(self.root_dir, 'pymarl/src/scripts/semdiv_eval_render.sh')
            eval_cmd = f'bash {script_path} {self.cuda_id} {self.tm_idx} {name} {tm_model_path} {tm_model_path} {n_traj} {self.env_idx} {self.env}'
            print(f"Running eval command: {eval_cmd}")
            ret_code = os.system(eval_cmd)
            if ret_code != 0:
                return 'bug', f'eval script failed with return code {ret_code >> 8}', None
            
            # 等待评估结果生成（最多等待5分钟）
            res_path = os.path.join(self.root_dir, f'pymarl/results/pymarl/{pymarl_env}/{name}/')
            max_wait_time = 300  # 5分钟
            wait_interval = 5  # 每5秒检查一次
            waited = 0
            while waited < max_wait_time:
                if os.path.exists(res_path) and len(os.listdir(res_path)) > 0:
                    break
                time.sleep(wait_interval)
                waited += wait_interval
                if waited % 30 == 0:  # 每30秒打印一次
                    print(f"Waiting for eval results... ({waited}/{max_wait_time}s)")
            
            if (not os.path.exists(res_path)) or len(os.listdir(res_path)) == 0:
                # 评估脚本未成功生成结果
                return 'bug', f'eval result dir not found after {waited}s: {os.path.abspath(res_path)}', None
            traj_path = os.path.join(res_path, os.listdir(res_path)[0], 'states.npy')
            res_path = os.path.join(res_path, os.listdir(res_path)[0], 'res.json')
            assert os.path.exists(traj_path)
            assert os.path.exists(res_path)
            
            if 'lbf' in self.env:
                with open(res_path) as f:
                    data = json.load(f)
                    foods_info = data['food_lst'] # ['A', 'B', ...]
                    trajs_info = np.load(traj_path) # [n_traj, seq_len, state_shape]
                    assert len(foods_info) == n_traj
                    assert trajs_info.shape[0] == n_traj
                self.traj_check_info = ''
                for i in range(n_traj):
                    food = foods_info[i]
                    traj = trajs_info[i]
                    agent1_coords = traj[:, -2:]
                    agent2_coords = traj[:, -4:-2]
                    non_zero_mask = ~np.all(traj == 0, axis=1)
                    agent1_valid_coords = agent1_coords[non_zero_mask].astype(int)
                    agent2_valid_coords = agent2_coords[non_zero_mask].astype(int)
                    traj_1 = ', '.join([f'({x[0]}, {x[1]})' for x in agent1_valid_coords])
                    traj_2 = ', '.join([f'({x[0]}, {x[1]})' for x in agent2_valid_coords])
                    if food == '':
                        self.traj_check_info += f'    Episode {i+1}: Agents failed to collect any food. '
                    else:
                        self.traj_check_info += f'    Episode {i+1}: Agents collected food {food}. '
                    self.traj_check_info += f'Trajectory of agent 1: {traj_1}. Trajectory of agent 2: {traj_2}.\n'
                input_check_prompt += self.traj_check_info
            elif 'pp' in self.env:
                with open(res_path) as f:
                    data = json.load(f)
                foods_info = data['food_lst']
                trajs_info = np.load(traj_path) # [n_traj, seq_len, state_shape]
                assert len(foods_info) == n_traj
                assert trajs_info.shape[0] == n_traj
                self.traj_check_info = ''
                for i in range(n_traj):
                    food = foods_info[i]
                    traj = trajs_info[i]
                    agent1_coords = traj[:, 0:0+2]
                    agent2_coords = traj[:, 2:2+2]
                    non_zero_mask = ~np.all(traj == 0, axis=1)
                    agent1_valid_coords = agent1_coords[non_zero_mask]
                    agent2_valid_coords = agent2_coords[non_zero_mask]
                    agent2_valid_coords += agent1_valid_coords
                    
                    agent1_valid_coords = agent1_valid_coords[::2]
                    agent2_valid_coords = agent2_valid_coords[::2]
                    
                    traj_1 = ', '.join([f'({x[0]:.2f}, {x[1]:.2f})' for x in agent1_valid_coords])
                    traj_2 = ', '.join([f'({x[0]:.2f}, {x[1]:.2f})' for x in agent2_valid_coords])
                    food2str = {
                        '0': 'S1',
                        '1': 'S2',
                        '2': 'R1',
                        '3': 'R2',
                        '4': 'R3',
                    }
                    if len(food) == 0:
                        self.traj_check_info += f'    Episode {i+1}: Predators failed to catch any prey. '
                    elif len(food) == 1:
                        self.traj_check_info += f'    Episode {i+1}: Predators caught prey {food2str[food]}. '
                    elif len(food) == 2:
                        self.traj_check_info += f'    Episode {i+1}: Predators caught prey {food2str[food[0]]} and {food2str[food[1]]}. '
                    self.traj_check_info += f'Trajectory of predator 1: {traj_1}. Trajectory of predator 2: {traj_2}.\n'
                input_check_prompt += self.traj_check_info
                    
            elif 'sc2' in self.env:
                with open(res_path) as f:
                    data = json.load(f)
                    kill_info = data['food_lst'] # ['A', 'B', ...]
                    trajs_info = np.load(traj_path) # [n_traj, seq_len, state_shape]
                    assert len(kill_info) == n_traj
                    assert trajs_info.shape[0] == n_traj
                self.traj_check_info = ''
                for i in range(n_traj):
                    kill = kill_info[i].replace("E", "")
                    traj = trajs_info[i]
                    init_state = traj[0]
                    agent1_coords = init_state[7 * 0 + 2 : 7 * 0 + 2 + 2]
                    agent2_coords = init_state[7 * 1 + 2 : 7 * 1 + 2 + 2]
                    enemy_A_coords = init_state[7 * 2 + 6 * 0 + 1 : 7 * 2 + 6 * 0 + 1 + 2]
                    enemy_B_coords = init_state[7 * 2 + 6 * 1 + 1 : 7 * 2 + 6 * 1 + 1 + 2]
                    enemy_C_coords = init_state[7 * 2 + 6 * 2 + 1 : 7 * 2 + 6 * 2 + 1 + 2]
                    enemy_D_coords = init_state[7 * 2 + 6 * 3 + 1 : 7 * 2 + 6 * 3 + 1 + 2]
                    agent1_coords = f"[{agent1_coords[0]:.2f}, {agent1_coords[1]:.2f}]"
                    agent2_coords = f"[{agent2_coords[0]:.2f}, {agent2_coords[1]:.2f}]"
                    enemy_A_coords = f"[{enemy_A_coords[0]:.2f}, {enemy_A_coords[1]:.2f}]"
                    enemy_B_coords = f"[{enemy_B_coords[0]:.2f}, {enemy_B_coords[1]:.2f}]"
                    enemy_C_coords = f"[{enemy_C_coords[0]:.2f}, {enemy_C_coords[1]:.2f}]"
                    enemy_D_coords = f"[{enemy_D_coords[0]:.2f}, {enemy_D_coords[1]:.2f}]"
                    if kill == '':
                        self.traj_check_info += f'    Episode {i+1}: Agents failed to kill any enemy. '
                    else:
                        self.traj_check_info += f'    Episode {i+1}: Agents killed enemy {kill}. '
                    self.traj_check_info += f'Initial postions: agent 1 at {agent1_coords}, agent 2 at {agent2_coords}, '
                    self.traj_check_info += f'enemy A at {enemy_A_coords}, enemy B at {enemy_B_coords}, enemy C at {enemy_C_coords}, enemy D at {enemy_D_coords}.\n'
                    first_attack_step = -1
                    for step in range(traj.shape[0]):
                        state = traj[step]
                        enemy_hp_lst = [state[7 * 2 + 6 * j] for j in range(4)]
                        if any(hp < 1.0 for hp in enemy_hp_lst):
                            first_attack_target = [f"enemy {chr(65 + j)}" for j, hp in enumerate(enemy_hp_lst) if hp < 1.0][0]
                            agent1_coords = f"[{state[7 * 0 + 2]:.2f}, {state[7 * 0 + 3]:.2f}]"
                            agent2_coords = f"[{state[7 * 1 + 2]:.2f}, {state[7 * 1 + 3]:.2f}]"
                            enemy_A_coords = f"[{state[7 * 2 + 6 * 0 + 1]:.2f}, {state[7 * 2 + 6 * 0 + 2]:.2f}]"
                            enemy_B_coords = f"[{state[7 * 2 + 6 * 1 + 1]:.2f}, {state[7 * 2 + 6 * 1 + 2]:.2f}]"
                            enemy_C_coords = f"[{state[7 * 2 + 6 * 2 + 1]:.2f}, {state[7 * 2 + 6 * 2 + 2]:.2f}]"
                            enemy_D_coords = f"[{state[7 * 2 + 6 * 3 + 1]:.2f}, {state[7 * 2 + 6 * 3 + 2]:.2f}]"
                            all_positions = {
                                "Agent1": agent1_coords,
                                "Agent2": agent2_coords,
                                "Enemy_A": enemy_A_coords,
                                "Enemy_B": enemy_B_coords,
                                "Enemy_C": enemy_C_coords,
                                "Enemy_D": enemy_D_coords
                            }
                            first_attack_step = step
                            break
                    if first_attack_step != -1:
                        self.traj_check_info += f'    The first attack targeted {first_attack_target}. '
                        self.traj_check_info += f'At this step, the positions were: agent 1 at {all_positions["Agent1"]}, agent 2 at {all_positions["Agent2"]}, '
                        self.traj_check_info += f'enemy A at {all_positions["Enemy_A"]}, enemy B at {all_positions["Enemy_B"]}, '
                        self.traj_check_info += f'enemy C at {all_positions["Enemy_C"]}, and enemy D at {all_positions["Enemy_D"]}.\n'
                    else:
                        self.traj_check_info += f'    No attack occurred during this episode.\n'
                input_check_prompt += self.traj_check_info
            else:
                assert 0

            input_check_prompt += ']\n'
            input_check_prompt += "Based on the information above, please review if the running behavior of the team aligns with the desired behavior or not. Think step by step, and tell us your answer. Make sure your output contains a string '::1::' if your answer is 'Yes' and contains a string '::0::' if your answer is 'No'."
            # print(input_check_prompt)
            self.traj_check_prompt = input_check_prompt
            answer_lst = []
            for _ in range(5):
                answer = -1
                while not answer in [0, 1]:
                    llm_check_output = self.llm.call_llm(input_check_prompt, big_model=self.big_model)
                    try:
                        answer = int(llm_check_output.split('::')[1])
                    except:
                        pass
                answer_lst.append((answer, llm_check_output))
            count_1 = sum(1 for answer, llm_check_output in answer_lst if answer == 1)
            count_0 = sum(1 for answer, llm_check_output in answer_lst if answer == 0)
            voting_answer = 1 if count_1 > count_0 else 0
            voting_llm_check_output = random.choice([llm_check_output for answer, llm_check_output in answer_lst if answer == voting_answer])
            self.traj_check_output = voting_llm_check_output
            if voting_answer == 0:
                clean_up(model_path)
                return 'misaligned', (input_check_prompt, voting_llm_check_output), None
        
        if self.involve_ego and self.tm_idx > 1:
            ego2info = {}
            ego_model_path = self.ego_library[self.tm_idx-1]['model_path'].split('pymarl/')[1]
            tm_model_path = model_path.split('pymarl/')[1]
            name = f'eval_{self.run_name}_tm_{self.tm_idx}_behavior_{self.behavior_idx}_attempt_{self.n_attempt_behavior}_with_ego_multi_head'
            script_path = os.path.join(self.root_dir, 'pymarl/src/scripts/semdiv_eval_xp.sh')
            cmd = f'bash {script_path} {self.cuda_id} {self.tm_idx} {name} {ego_model_path} {tm_model_path} {self.env_idx} {self.env}'
            print(cmd)
            ret_code = os.system(cmd)
            if ret_code != 0:
                return 'bug', f'eval_xp script failed with return code {ret_code >> 8}', None
            
            # 等待评估结果生成（最多等待5分钟）
            eval_result_dir = os.path.join(self.root_dir, f'pymarl/results/pymarl/{pymarl_env}', name)
            max_wait_time = 300  # 5分钟
            wait_interval = 5  # 每5秒检查一次
            waited = 0
            while waited < max_wait_time:
                if os.path.exists(eval_result_dir) and len(os.listdir(eval_result_dir)) > 0:
                    break
                time.sleep(wait_interval)
                waited += wait_interval
                if waited % 30 == 0:  # 每30秒打印一次
                    print(f"Waiting for eval_xp results... ({waited}/{max_wait_time}s)")
            
            if (not os.path.exists(eval_result_dir)) or len(os.listdir(eval_result_dir)) == 0:
                # 评估脚本未成功生成结果
                return 'bug', f'eval_xp result dir not found after {waited}s: {os.path.abspath(eval_result_dir)}', None
            
            assert len(os.listdir(eval_result_dir)) == 1
            res = os.listdir(eval_result_dir)[0]
            json_path = os.path.join(eval_result_dir, res)
            for head_res_file in os.listdir(json_path): # 'res_1.json'
                if 'res_' not in head_res_file or '.json' not in head_res_file:
                    continue
                res_path = os.path.join(json_path, head_res_file)
                with open(res_path) as f:
                    data = json.load(f)
                head_id = int(head_res_file.split('res_')[1].split('.json')[0])
                ego_idx = head_id + 1
                ego2info[ego_idx] = {'ret': data['ret'], 'ret_additional': data['ret_additional'], 'is_similar': False, 'behavior': self.ego_library[ego_idx]['behavior']}
            baseline_ret = performance
            baseline_ret_additional = np.mean(values[-5:])
            ego2info['baseline_ret'] = baseline_ret
            ego2info['baseline_ret_additional'] = baseline_ret_additional
            for ego_idx, ego_info in ego2info.items():
                if type(ego_info) != type({}):
                    continue
                if (baseline_ret - ego_info['ret']) / (abs(baseline) + 1e-5) < self.ego_performance_tolerance and \
                    (baseline_ret_additional - ego_info['ret_additional']) / (abs(baseline_ret_additional) + 1e-5) < self.ego_performance_additional_tolerance:
                    ego_info['is_similar'] = True
            for ego_idx, ego_info in ego2info.items():
                if type(ego_info) != type({}):
                    continue
                if ego_info['is_similar'] and self.do_similarity_check:
                    clean_up(model_path)
                    return 'similar', ego2info, None
    
        return 'success', f'{str(performance)}_{str(performance_additional)}', model_path

    def check_status_football(self):
        path_ = os.path.join(self.root_dir, f'HARL/examples/results/football/{self.map}/mappo/{self.run_name}/')
        assert os.path.exists(path_), path_
        folders = [x for x in os.listdir(path_) if os.path.isdir(os.path.join(path_, x))]
        assert len(folders) == 1, os.listdir(path_)
        path = os.path.join(path_, folders[0])
        model_path = os.path.join(path, 'models')

        def clean_up(_path):
            os.system(f"pkill -f '{self.run_name}'")
            if _path is not None and os.path.exists(_path):
                shutil.rmtree(_path)

        if not (os.path.exists(path) and 'models' in os.listdir(path) and 'actor_agent0_-1.pt' in os.listdir(model_path) and 'summary.json' in os.listdir(os.path.join(path, 'logs'))):
            clean_up(path)
            message = ""
            try:
                with open(os.path.join(path_, '1.out'), 'r') as file:
                    message = file.read()
                    message = message.split('Traceback (most recent call last):')[-2].split('File ')[-1]
            except:
                pass
            return 'bug', message, None

        tb_path = os.path.join(path, 'logs', 'eval_score_rate', 'eval_score_rate')
        tb_path = os.path.join(tb_path, os.listdir(tb_path)[0])
        ea = event_accumulator.EventAccumulator(tb_path)
        ea.Reload()
        
        scalar_data = ea.scalars.Items('eval_score_rate')
        try:
            values = [scalar_data[i][2] for i in range(len(scalar_data))]
        except:
            values = [scalar_data[i].value for i in range(len(scalar_data))]
        performance = np.mean(values[-5:])
        prior_performance = 1.0
        self.baseline = prior_performance
        baseline = self.baseline
        if performance < baseline * (1 - self.tm_performance_tolerance):
            clean_up(path)
            return 'failed', str(performance), None
        
        tb_path = os.path.join(path, 'logs', 'eval_average_episode_rewards_additional', 'eval_average_episode_rewards_additional')
        tb_path = os.path.join(tb_path, os.listdir(tb_path)[0])
        ea = event_accumulator.EventAccumulator(tb_path)
        ea.Reload()
        scalar_data = ea.scalars.Items('eval_average_episode_rewards_additional')
        try:
            values = [scalar_data[i][2] for i in range(len(scalar_data))]
        except:
            values = [scalar_data[i].value for i in range(len(scalar_data))]
        performance_additional = np.mean(values[-5:])
        performance_additional_init = np.mean(values[:5])
        if (performance_additional - performance_additional_init) / (abs(performance_additional_init) + 1e-5) < 0.1: # no improvement
            clean_up(path)
            return 'constant', str(performance_additional_init), None
        
        if self.do_traj_check:
            behavior = self.behavior_library[self.behavior_idx]['behavior']
            input_check_prompt = prompt_basic_env + prompt_code + f"\nWe tried to train a team with this specific cooperation behavior:\n---\n{behavior}\n---\n"
            input_check_prompt += "After training the team with this reward function, we ran it for an episode: [\n"
            tm_model_path = model_path.split('HARL/examples/')[1]
            name = f'eval_{self.run_name}_tm_{self.tm_idx}_behavior_{self.behavior_idx}_attempt_{self.n_attempt_behavior}_sp'
            use_render = True
            head_id = -1
            head_path = ""
            script_path = os.path.join(self.root_dir, 'HARL/examples/eval_semdiv.sh')
            cmd = f'bash {script_path} {self.cuda_id} {self.tm_idx} {name} {self.env_idx} {tm_model_path} {tm_model_path} {use_render} {head_id} {head_path}'
            print(cmd)
            os.system(cmd)
            res_path = os.path.join(self.root_dir, f'HARL/examples/results/{name}/res.json')
            assert os.path.exists(res_path)
            with open(res_path) as f:
                data = json.load(f)
                pass_history = data['pass_history']
                score = data['score']
                score_info = data['score_info']
            self.traj_check_info = f'    In this episode, '
            if len(pass_history) == 0:
                self.traj_check_info += f"players did not pass the ball, "
            if len(pass_history) == 1:
                self.traj_check_info += f"{pass_history[0][0]} passed to {pass_history[0][1]}, "
            else:
                for i, passing in enumerate(pass_history):
                    self.traj_check_info += f"({i+1}) {passing[0]} passed to {passing[1]}, "
            if score:
                goal_player = score_info['goal_player']
                ball_position = ','.join([str(round(x, 2)) for x in score_info['ball_position']])
                Turing_position = ','.join([str(round(x, 2)) for x in score_info['Turing_position']])
                Johnson_position = ','.join([str(round(x, 2)) for x in score_info['Johnson_position']])
                Meitner_position = ','.join([str(round(x, 2)) for x in score_info['Meitner_position']])
                self.traj_check_info += f"and finally successfully scored a goal. The player who scored the goal is {goal_player}. At the moment of scoring, the ball is at ({ball_position}), Turing is at ({Turing_position}), Johnson is at ({Johnson_position}), Meitner is at ({Meitner_position}).\n"
            else:
                self.traj_check_info += 'but failed to score a goal.\n'
            input_check_prompt += self.traj_check_info
            
            input_check_prompt += ']\n'
            input_check_prompt += "Based on the information above, please review if the running behavior of the team aligns with the desired behavior or not. Think step by step, and tell us your answer. Make sure your output contains a string '::1::' if your answer is 'Yes' and contains a string '::0::' if your answer is 'No'."
            # print(input_check_prompt)
            self.traj_check_prompt = input_check_prompt
            answer_lst = []
            for _ in range(5):
                answer = -1
                while not answer in [0, 1]:
                    llm_check_output = self.llm.call_llm(input_check_prompt, big_model=self.big_model)
                    try:
                        answer = int(llm_check_output.split('::')[1])
                    except:
                        pass
                answer_lst.append((answer, llm_check_output))
            count_1 = sum(1 for answer, llm_check_output in answer_lst if answer == 1)
            count_0 = sum(1 for answer, llm_check_output in answer_lst if answer == 0)
            voting_answer = 1 if count_1 > count_0 else 0
            voting_llm_check_output = random.choice([llm_check_output for answer, llm_check_output in answer_lst if answer == voting_answer])
            self.traj_check_output = voting_llm_check_output
            if voting_answer == 0:
                clean_up(path)
                return 'misaligned', (input_check_prompt, voting_llm_check_output), None
        
        if self.involve_ego and self.tm_idx > 1:
            ego2info = {}
            feature_extractor_path = self.ego_library[self.tm_idx-1]['model_path'].split('HARL/examples/')[1] # actually feature extractor
            tm_model_path = model_path.split('HARL/examples/')[1]
            for ego_idx, ego_info in self.ego_library.items():
                if len(ego_info) == 0:
                    continue
                head_path = ego_info['model_path'].split('HARL/examples/')[1]
                name = f'eval_{self.run_name}_tm_{self.tm_idx}_behavior_{self.behavior_idx}_attempt_{self.n_attempt_behavior}_ego_{ego_idx}'
                use_render = False
                head_id = 1 - self.load_agent_id
                agent0_model_path, agent1_model_path = (tm_model_path, feature_extractor_path) if self.load_agent_id == 0 else (feature_extractor_path, tm_model_path)
                script_path = os.path.join(self.root_dir, 'HARL/examples/eval_semdiv.sh')
                cmd = f'bash {script_path} {self.cuda_id} {self.tm_idx} {name} {self.env_idx} {agent0_model_path} {agent1_model_path} {use_render} {head_id} {head_path}'
                print(cmd)
                os.system(cmd)
                res_path = os.path.join(self.root_dir, f'HARL/examples/results/{name}/res.json')
                assert os.path.exists(res_path), res_path
                with open(res_path) as f:
                    data = json.load(f)
                ego2info[ego_idx] = {'ret': data['eval_score_rate'], 'ret_additional': data['eval_average_episode_rewards_additional'], 'is_similar': False, 'behavior': ego_info['behavior']}
            baseline_ret = performance
            baseline_ret_additional = np.mean(values[-5:])
            ego2info['baseline_ret'] = baseline_ret
            ego2info['baseline_ret_additional'] = baseline_ret_additional
            for ego_idx, ego_info in ego2info.items():
                if type(ego_info) != type({}):
                    continue
                if (baseline_ret - ego_info['ret']) / (abs(baseline) + 1e-5) < self.ego_performance_tolerance and \
                    (baseline_ret_additional - ego_info['ret_additional']) / (abs(baseline_ret_additional) + 1e-5) < self.ego_performance_additional_tolerance:
                    ego_info['is_similar'] = True
            for ego_idx, ego_info in ego2info.items():
                if type(ego_info) != type({}):
                    continue
                if ego_info['is_similar'] and self.do_similarity_check:
                    clean_up(path)
                    return 'similar', ego2info, None
    
        return 'success', f'{str(performance)}_{str(performance_additional)}', model_path

if __name__ == "__main__":
    # 使用脚本所在目录来构造路径，避免依赖当前工作目录
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../SemDiv/language
    _ROOT_DIR = os.path.dirname(_THIS_DIR)                          # .../SemDiv
    
    timing = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(timing)
    
    env = 'lbf'
    # env = 'pp'
    # env = 'sc2'
    # env = 'football:academy_pass_and_shoot_with_keeper'

    if env == 'lbf':
        from prompt_text_lbf import *
        env_file_init = os.path.join(
            _ROOT_DIR,
            "pymarl/src/envs/lb-foraging/lbforaging/foraging/environment_init.py",
        )
        env_file = os.path.join(
            _ROOT_DIR,
            "pymarl/src/envs/lb-foraging/lbforaging/foraging/environment.py",
        )
    elif env == 'pp':
        from prompt_text_pp import *
        env_file_init = os.path.join(
            _ROOT_DIR,
            "pymarl/src/envs/mpe/multi_agent_particle/mpe/scenarios/simple_tag-init.py",
        )
        env_file = os.path.join(
            _ROOT_DIR,
            "pymarl/src/envs/mpe/multi_agent_particle/mpe/scenarios/simple_tag-1.py",
        )
    elif env == 'sc2':
        from prompt_text_sc2 import *
        env_file_init = os.path.join(
            _ROOT_DIR,
            "pymarl/src/envs/smacv2/smacv2/env/starcraft2/additional_reward_init.py",
        )
        env_file = os.path.join(
            _ROOT_DIR,
            "pymarl/src/envs/smacv2/smacv2/env/starcraft2/additional_reward_1.py",
        )
    elif 'football' in env:
        from prompt_text_football import *
        env_file_init = os.path.join(
            _ROOT_DIR,
            "football/gfootball/env/reward_wrapper_init.py",
        )
        env_file = os.path.join(
            _ROOT_DIR,
            "football/gfootball/env/reward_wrapper_1.py",
        )
    else:
        assert 0

    with open(env_file_init, "r") as source_file:
        content = source_file.read()
    with open(env_file, "w") as target_file:
        target_file.write(content)
    semdiv = SemDiv(env, env_file, timing)
    semdiv.run()
