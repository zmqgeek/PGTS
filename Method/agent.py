import torch
import torch.nn.functional as F
import os
from gallina_astactic import GallinaTermParser
from utils import SexpCache, log
from eval_env import FileEnv
import re
import pickle
from progressbar import ProgressBar
from glob import glob
import json
from random import random
import pdb
from hashlib import sha1
import gc
from copy import deepcopy
from time import time
from search_tree import ProofSearchTree


def action_seq_loss(logits_batch, actions_batch, opts):
    assert len(logits_batch) == len(actions_batch)
    loss = 0
    for logits, actions in zip(logits_batch, actions_batch):
        length = min(logits.shape[0], actions.shape[0])
        loss += F.cross_entropy(logits[:length], actions[:length].to(opts.device))
    loss /= len(logits_batch)
    return loss


# merge this with extract_proof_steps.py
term_parser = GallinaTermParser(caching=True)
sexp_cache = SexpCache("../sexp_cache", readonly=True)


def filter_env(env):
    "Get the last 10 toplevel constants"
    filtered_env = []
    toplevel_consts = [
        const for const in env["constants"] if const["qualid"].startswith("SerTop")
    ]
    for const in toplevel_consts[-10:]:
        ast = sexp_cache[const["sexp"]]
        filtered_env.append({"qualid": const["qualid"], "ast": term_parser.parse(ast)})
    return filtered_env


def parse_goal(g):
    goal = {"id": g["id"], "text": g["type"], "ast": term_parser.parse(g["sexp"])}
    local_context = []
    for i, h in enumerate(g["hypotheses"]):
        for ident in h["idents"]:
            local_context.append(
                {"ident": ident, "text": h["type"], "ast": term_parser.parse(h["sexp"])}
            )
    return local_context, goal["ast"]

def load_from_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def get_tactic_name(tactic):
    return tactic.split()[0]

def extract_base_tactic(tactic):
    """
    提取策略的基本名称，去除参数。
    输入：单个策略字符串（如 "intros x y" 或 "apply H with (n := 3)"）
    输出：无参数策略名称（如 "intros" 或 "apply"）
    """
    # 使用正则表达式提取第一个单词作为策略名称
    match = re.match(r"^\s*(?:by\s+)?(\w+)", tactic)
    # match = re.match(r"^\s*(\w+)", tactic)
    if match:
        return match.group(1)
    return tactic 

def sort_tactics(prev_tactic, candidates, pattern_support):
    prev_tactic_name = get_tactic_name(prev_tactic)
    # prev_tactic_name = extract_base_tactic(prev_tactic)
    candidate_with_index = [(tactic, i) for i, tactic in enumerate(candidates)]
    matching = []
    non_matching = []
    
    for tactic, index in candidate_with_index:
        tactic_name = get_tactic_name(tactic)
        # tactic_name = extract_base_tactic(tactic)
        if (prev_tactic_name, tactic_name) in pattern_support:
            matching.append((tactic, pattern_support[(prev_tactic_name, tactic_name)], index)) 
        else:
            non_matching.append((tactic, index))
    matching.sort(key=lambda x: (x[1], x[2]))
    non_matching.sort(key=lambda x: x[1])

    sorted_matching = [tactic for tactic, _, _ in matching]
    sorted_non_matching = [tactic for tactic, _ in non_matching]

    combined = sorted_non_matching + sorted_matching
    return combined 

def print_single_goal(g):
    for h in g["hypotheses"]:
        for ident in h["idents"]:
            print("\t%s: %s" % (ident, h["type"]))
    print("---------------")
    print("\t%s" % g["type"])
    print("##########")


def print_goals(obs):
    if "fg_goals" not in obs:
        print("##########")
        return
    print("########## fg_goals ##########")
    for g in obs["fg_goals"]:
        print_single_goal(g)
    print("########## bg_goals ##########")
    for g in obs["bg_goals"]:
        print_single_goal(g)
    print("########## shelved_goals ##########")
    for g in obs["shelved_goals"]:
        print_single_goal(g)
    print("########## given_up_goals ##########")
    for g in obs["given_up_goals"]:
        print_single_goal(g)


def get_goal_signature(goal):
    sexp = goal["sexp"] + "".join([h["sexp"] for h in goal["hypotheses"]])
    return sha1(sexp.encode("utf-8")).hexdigest()


class Agent:
    def __init__(self, model, optimizer, dataloader, opts):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.opts = opts
        self.projs_split = json.load(open(opts.projs_split))

    def train(self, n_epoch):
        self.model.train()
        log("training with teacher forcing %f.." % self.opts.teacher_forcing)

        bar = ProgressBar(max_value=len(self.dataloader["train"]))
        for i, data_batch in enumerate(self.dataloader["train"]):
            use_teacher_forcing = random() < self.opts.teacher_forcing
            asts, loss = self.model(
                data_batch["env"],
                data_batch["local_context"],
                data_batch["goal"],
                data_batch["tactic_actions"],
                use_teacher_forcing,
            )
            log(
                "\nteacher forcing = %s, loss = %f"
                % (str(use_teacher_forcing), loss.item())
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            gc.collect()
            bar.update(i)
            if self.opts.smoke and i == 11:
                break

        log("\ntraining losses: %f" % loss)

    def valid(self, n_epoch):
        self.model.eval()
        log("validating..")
        loss_avg = 0
        predictions = []
        num_correct = 0
        bar = ProgressBar(max_value=len(self.dataloader["valid"]))

        for i, data_batch in enumerate(self.dataloader["valid"]):
            asts, loss = self.model(
                data_batch["env"],
                data_batch["local_context"],
                data_batch["goal"],
                data_batch["tactic_actions"],
                False,
            )
            loss_avg += loss.item()

            for n in range(len(data_batch["file"])):
                tac_gt = data_batch["tactic_str"][n]
                tac_pred = asts[n].to_tokens()
                if tac_gt.replace(" ", "") == tac_pred.replace(" ", ""):
                    num_correct += 1
                predictions.append(
                    {
                        "file_name": data_batch["file"][n],
                        "proof_name": data_batch["proof_name"][n],
                        "n_step": data_batch["n_step"][n],
                        "tac_gt": tac_gt,
                        "tac_pred": tac_pred,
                    }
                )
            gc.collect()
            bar.update(i)
            if self.opts.smoke and i == 11:
                break

        pickle.dump(
            predictions,
            open(
                os.path.join(
                    self.opts.log_dir, "predictions/pred_%03d.pickle" % n_epoch
                ),
                "wb",
            ),
        )

        loss_avg /= len(self.dataloader["valid"])
        log("\nvalidation losses: %f" % loss_avg)
        acc = num_correct / len(predictions)
        log("validation accuracy: %f" % acc)
        return loss_avg

    def evaluate(self, filename, proof_name=None):
        if self.model is not None:
            self.model.eval()

        if "hammer" in self.opts.method:
            for atp in ["Vampire", "Z3", "CVC4", "Eprover"]:
                if ("hammer_" + atp) in self.opts.method:
                    with_hammer = atp
                    self.opts.method = self.opts.method.replace(
                        "hammer_" + atp, "hammer"
                    )
                    break
            else:
                with_hammer = "All"
        else:
            with_hammer = None
        assert "hammer_" not in self.opts.method
        hammer_timeout = (
            self.opts.hammer_timeout
            if "ours" in self.opts.method
            else self.opts.timeout
        )

        with FileEnv(
            filename,
            self.opts.max_num_tactics,
            self.opts.timeout,
            with_hammer=with_hammer,
            hammer_timeout=hammer_timeout,
        ) as file_env:
            results = []
            for proof_env in file_env:  # start a proof
                if proof_name is not None and proof_env.proof["name"] != proof_name:
                    continue
                print("proof: ", proof_env.proof["name"])
                # print('cuda memory allocated before proof: ', torch.cuda.memory_allocated(self.opts.device), file=sys.stderr)
                success, proof_pred, time, num_tactics,root = self.prove(proof_env)
                results.append(
                    {
                        "filename": filename,
                        "proof_name": proof_env.proof["name"],
                        "success": success,
                        "proof_gt": [
                            step["command"][0]
                            for step in proof_env.proof["steps"]
                            if step["command"][1] != "VernacEndProof"
                        ],
                        "proof_pred": proof_pred,
                        "time": time,
                        "num_tactics": num_tactics,
                        "proof_tree": root.to_dict(),
                    }
                )
                if proof_name is not None:
                    break
        return results

    def prove_one_tactic(self, proof_env, tac):
        obs = proof_env.init()
        print_goals(obs)
        obs = proof_env.step(tac + ".")
        print(obs["result"])
        print_goals(obs)
        root = ProofSearchTree(proof_env.proof['name'])
        time = self.opts.timeout - obs["time_left"]
        if obs["result"] == "SUCCESS":
            return True, [tac], time, 1, root
        else:
            return False, [tac], time, 1, root

    def prove(self, proof_env):
        "prove a theorem interactively"
        if "ours" not in self.opts.method:  # auto, hammer, etc.
            return self.prove_one_tactic(proof_env, self.opts.method)

        m = re.fullmatch(
            r"ours\+(?P<auto_tac>\w+)", self.opts.method
        )  # ours+auto/hammer/etc.
        if m is not None:
            tac_template = "%s."
            #tac_template = m["auto_tac"] + "; %s."
        else:
            tac_template = "%s."

        return self.prove_DFS(proof_env, tac_template)

    def prove_DFS(self, proof_env, tac_template):
        obs = proof_env.init()
        env = filter_env(obs["env"])
        first_goal_signatures = {get_goal_signature(obs["fg_goals"][0])}

        loaded_pattern_support = load_from_pickle('pattern_support.pickle')

        # initialize the stack
        local_context, goal = parse_goal(obs["fg_goals"][0])
        tactics = self.model.beam_search(env, local_context, goal)
        stack = [[tac_template % tac.to_tokens() for tac in tactics[::-1]]]

        script = []

        # initialize proof tree
        root = ProofSearchTree(proof_env.proof['name'])
        current_node = root

        # depth-first search starting from the trace
        while stack != [[]]:
            # print('stack: ', stack)
            # pick a tactic
            if stack[-1] == []:  # all candidate have been tried, backtrack
                stack.pop()
                script.pop()
                proof_env.step("Undo.")
                current_node.current_proof_state = "Backtraced"
                current_node = current_node.parent
                continue
            else:
                tac = stack[-1].pop()

            obs = proof_env.step(tac)
            print(obs["result"])
            print_goals(obs)

            if obs["result"] == "SUCCESS":
                script.append(tac)
                current_node.add_child(ProofSearchTree(obs["result"], used_tactic=tac))
                time = self.opts.timeout - obs["time_left"]
                num_tactics = self.opts.max_num_tactics - obs["num_tactics_left"]
                return True, script, time, num_tactics,root
            elif obs["result"] in ["MAX_NUM_TACTICS_REACHED", "MAX_TIME_REACHED"]:
                time = self.opts.timeout - obs["time_left"]
                num_tactics = self.opts.max_num_tactics - obs["num_tactics_left"]
                return False, script, time, num_tactics,root
            elif obs["result"] in ["ERROR"]:
                current_node.add_child(ProofSearchTree(obs["result"], used_tactic=tac))
                continue
            else:
                assert obs["result"] == "PROVING"
                script.append(tac)
                new_node = ProofSearchTree(obs["result"], used_tactic=tac)
                current_node.add_child(new_node)
                current_node = new_node

                sig = get_goal_signature(obs["fg_goals"][0])
                if sig in first_goal_signatures or len(script) >= self.opts.depth_limit:
                    if sig in first_goal_signatures:
                        current_node.current_proof_state = "GoalDouble"
                    else:
                        current_node.current_proof_state = "DepthLimit"
                    current_node = current_node.parent
                    proof_env.step("Undo.")
                    script.pop()
                    continue
                first_goal_signatures.add(sig)
                local_context, goal = parse_goal(obs["fg_goals"][0])
                prev_tactic = script[-1]
                tactics = self.model.beam_search(env, local_context, goal)
                candidate_tactics = [tac_template % tac.to_tokens() for tac in tactics[::-1]]
                sorted_tactics = sort_tactics(prev_tactic, candidate_tactics, loaded_pattern_support)
                stack.append(sorted_tactics)
                # stack.append([tac_template % tac.to_tokens() for tac in tactics[::-1]])

        obs = proof_env.step("Admitted.")
        print(obs["result"])
        time = self.opts.timeout - obs["time_left"]
        num_tactics = self.opts.max_num_tactics - obs["num_tactics_left"]
        return False, script, time, num_tactics,root

    def prove_IDDFS(self, proof_env, tac_template):
        obs = proof_env.init()
        env = filter_env(obs["env"])
        first_goal_signatures = {get_goal_signature(obs["fg_goals"][0])}
        depth_limit = self.opts.depth_limit
        traces = [[]]

        # iterative deepening depth-first search
        while traces != []:
            # depth-first search with depth_limit
            new_traces = []  # the newly-discovered truncated proofs
            for script in traces:
                # execute the tactics in the trace
                for tac in script:
                    obs = proof_env.step(tac)
                print(obs["result"])
                print_goals(obs)
                if obs["result"] != "PROVING":
                    assert obs["result"] in [
                        "MAX_NUM_TACTICS_REACHED",
                        "MAX_TIME_REACHED",
                    ]
                    time = self.opts.timeout - obs["time_left"]
                    num_tactics = self.opts.max_num_tactics - obs["num_tactics_left"]
                    return False, script, time, num_tactics
                # initialize the stack
                local_context, goal = parse_goal(obs["fg_goals"][0])
                tactics = self.model.beam_search(env, local_context, goal)
                stack = [[tac_template % tac.to_tokens() for tac in tactics[::-1]]]

                # depth-first search starting from the trace
                while stack != [[]]:
                    print("stack: ", stack)
                    # pick a tactic
                    if stack[-1] == []:  # all candidate have been tried, backtrack
                        stack.pop()
                        script.pop()
                        proof_env.step("Undo.")
                        continue
                    else:
                        tac = stack[-1].pop()

                    obs = proof_env.step(tac)
                    print(obs["result"])
                    print_goals(obs)

                    if obs["result"] == "SUCCESS":
                        script.append(tac)
                        time = self.opts.timeout - obs["time_left"]
                        num_tactics = (
                            self.opts.max_num_tactics - obs["num_tactics_left"]
                        )
                        return True, script, time, num_tactics
                    elif obs["result"] in [
                        "MAX_NUM_TACTICS_REACHED",
                        "MAX_TIME_REACHED",
                    ]:
                        time = self.opts.timeout - obs["time_left"]
                        num_tactics = (
                            self.opts.max_num_tactics - obs["num_tactics_left"]
                        )
                        return False, script, time, num_tactics
                    elif obs["result"] in ["ERROR"]:
                        continue
                    else:
                        assert obs["result"] == "PROVING"
                        script.append(tac)
                        sig = get_goal_signature(obs["fg_goals"][0])
                        if sig in first_goal_signatures or len(script) >= depth_limit:
                            if (
                                len(script) >= depth_limit
                                and sig not in first_goal_signatures
                            ):
                                new_traces.append(deepcopy(script))
                            proof_env.step("Undo.")
                            script.pop()
                            continue
                        first_goal_signatures.add(sig)
                        local_context, goal = parse_goal(obs["fg_goals"][0])
                        tactics = self.model.beam_search(env, local_context, goal)
                        stack.append(
                            [tac_template % tac.to_tokens() for tac in tactics[::-1]]
                        )

                proof_env.step("Restart.")
                gc.collect()

            depth_limit *= 2
            traces = new_traces

        obs = proof_env.step("Admitted.")
        print(obs["result"])
        time = self.opts.timeout - obs["time_left"]
        num_tactics = self.opts.max_num_tactics - obs["num_tactics_left"]
        return False, script, time, num_tactics

    def save(self, n_epoch, dirname):
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "n_epoch": n_epoch,
                "optimizer": self.optimizer.state_dict(),
            },
            os.path.join(dirname, "model_%03d.pth" % n_epoch),
        )
