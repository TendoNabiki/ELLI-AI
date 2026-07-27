from gettext import install

import fsspec
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datasets import load_dataset
from transformers import (AutoTokenizer, LlamaConfig, LlamaForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments)

from datasets import load_dataset

dl = load_dataset("m-a-p/SuperGPQA")

dl_train = dl["train"].to_pandas()

dl_train.head()
dl_train.isna().sum()
output_filename4 = 'dl_final.csv'

dl_final = dl_train.drop(columns = ["uuid", "options", "answer_letter", "discipline", 
"field", "subfield", "difficulty", "is_calculation"])

dl_final.to_csv(output_filename4, index=False)

print(dl_final.head())

from datasets import load_dataset

ds = load_dataset("mpingale/mental-health-chat-dataset")

ds_train = ds["train"].to_pandas()

ds_train.head()
ds_train.isna().sum()
ds_final = ds_train.dropna(subset=['questionText']).copy()
ds_final = ds_final.dropna(subset=['answerText']).copy()
ds_final.isna().sum()
ds_final = ds_final.drop(columns=["questionID", "questionTitle", "questionLink", "topic",
"therapistInfo", "therapistURL", "upvotes", "views", "text"], errors='ignore')

ds_final = ds_final.rename(columns={'questionText': 'question', 'answerText': 'answer'})
ds_final.head()

output_filename = 'ds_final.csv'
ds_final.to_csv(output_filename, index=False)
print(ds_final.head())

from datasets import load_dataset

dt = load_dataset("open-thoughts/OpenThoughts-Agent-SFT-100K")

dt_train = dt["train"].to_pandas()

dt_train.head()
dt_train.isna().sum()
dt_final = dt_train.dropna(subset=['result']).copy()
dt_final = dt_final.dropna(subset=['trace_source']).copy()

dt_final = dt_final.drop(columns=["agent", "date", "episode", "model", 
"model_provider", "run_id", "task", "trace_source", "trial_name"])

output_filename1 = 'dt_final.csv'
dt_final.to_csv(output_filename1, index=False)

print(dt_final.head())

from datasets import load_dataset

dk = load_dataset("uw-math-ai/math-graph", "paper_lean_community")

dk_train = dk["train"].to_pandas()

dk_train.head()
dk_train.isna().sum()
dk_final = dk_train.drop(columns=["paper_id", "kind", "source", 
"authors", "repo_slug", "categories", "updated_at", "branch", "src_path"])

dk_final = dk_final.rename(columns={'title': 'question', 'answer': 'url'})
print(dk_final.head())

output_filename2 = 'dk_final.csv'
dk_final.to_csv(output_filename2, index=False)

from datasets import load_dataset

dn = load_dataset("fka/prompts.chat")

dn_train = dn["train"].to_pandas()

dn_train.head()
dn_train.isna().sum()

dn_final = dn_train.drop(columns=["for_devs", "type", "contributor"])
dn_final = dn_final.rename(columns={'act': 'question', 'prompt': 'answer'})
print(dn_final.head())

output_filename3 = 'dn_final.csv'
dn_final.to_csv(output_filename3, index=False)




from datasets import load_dataset

da = load_dataset("WithinUsAI/claude_mythos_distilled_25k")

da_train = da["train"].to_pandas()

da_train.head()
da_train.isna().sum()

da_train.head()

da_final = da_train.drop(columns=["id", "source", "timestamp"])
da_final = da_final.rename(columns={"messages" : "question", "category" : "answer"})

da_final.head()

da_final_json_filename = "da_final.json"
da_final.to_json(da_final_json_filename, orient = "records", indent = 4)

from datasets import load_dataset

dr = load_dataset("IntelligenceLab/Long-Horizon-Terminal-Bench")
dr_test = dr["test"].to_pandas()
dr_test.head()

dr_final = dr_test.drop(columns=["task_id", "name", "category", "difficulty", "keywords", "docker_image", "allow_internet", "cpus", "memory_mb", "gpus", "agent_timeout_min", "expert_time_estimate_min"])
dr_final = dr_final.rename(columns = {"description" : "question", "instruction" : "answer"})

dr_final.head()

dr_final.isna().sum()

dr_final_json_filename = 'dr_final.json'
dr_final.to_json(dr_final_json_filename, orient='records', indent=4)

from datasets import load_dataset

db = load_dataset("google/civil_comments")

db_train = db["train"].to_pandas()
db_train.head()

db_final = db_train.drop(columns=[ "severe_toxicity", "obscene", "identity_attack", "insult", "threat", "sexual_explicit"])
db_final = db_final.rename(columns = {"text" : "question", "toxicity" : "answer"})

db_final.head()

db_final_json_filename = "db_final.json"
db_final.to_json(db_final_json_filename, orient='records', indent=4)

from datasets import load_dataset

dc = load_dataset("microsoft/MeetingBank-QA-Summary")

dc_test = dc["test"].to_pandas()
dc_test.head()

dc_test.isna().sum()

dc_final = dc_test.drop(columns=["idx", "QA_pairs", "gpt4_summary"])
dc_final = dc_final.rename(columns={"prompt" : "question", "summary" : "answer"})

dc_final.head()

dc_final_json_filename = "dc_final.json"
dc_final.to_json(dc_final_json_filename, orient='records', indent=4)

from datasets import load_dataset

dd = load_dataset("SupraLabs/reasoning-summaries-61k")

dd_train = dd["train"].to_pandas()
dd_train.head()

dd_train.isna().sum()

dd_final = dd_train.rename(columns={"user" : "question", "assistant" : "answer"})

dd_final.head()

dd_final_json_filename = "dd_final.json"
dd_final.to_json(dd_final_json_filename, orient='records', indent=4)

from datasets import load_dataset

de = load_dataset("MatSciBench/MatSciBench")

de_test = de["test"].to_pandas()
de_test.head()

de_test.isna().sum()

de_final = de_test[['question', 'answer']]
de_final.head()

de_final = de_final.dropna(subset=['answer'])

de_final_json_filename = "de_final.json"
de_final.to_json(de_final_json_filename, orient='records', indent=4)

from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
df = load_dataset("snorkelai/Multi-Turn-Insurance-Underwriting")

df_train = df["train"].to_pandas()
df_train.head()

df_train.isna().sum()

df_final = df_train[["reference answer", "correct"]]
df_final = df_final.rename(columns={"reference answer" : "question", "correct" : "answer"})

df_final.head()

df_final_json_filename = "df_final.json"
df_final.to_json(df_final_json_filename, orient='records', indent=4)

from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
dg = load_dataset("theprint/Esoteric-Math")

dg_train = dg["train"].to_pandas()
dg_train.head()

dg_train.isna().sum()

dg_final = dg_train[["question", "answer"]]

dg_final.head()

dg_final_json_filename = "dg_final.json"
dg_final.to_json(dg_final_json_filename, orient='records', indent=4)

from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
dh = load_dataset("galileo-ai/agent-leaderboard-v2", "adaptive_tool_use")

dh_train = dh["telecom"].to_pandas()

dh_train.head()

dh_train.isna().sum()

dh_final = dh_train.drop(columns=["persona_index"])
dh_final = dh_final.rename(columns={"first_message" : "question", "user_goals" : "answer"})

dh_final.head()

dh_final_json_filename = "dh_final.json"
dh_final.to_json(dh_final_json_filename, orient='records', indent=4)

from datasets import load_dataset

di = load_dataset("MathArena/arxivmath-training")

di_train = di["train"].to_pandas()
di_train.head()

di_train.isna().sum()

di_final = di_train[["question", "answer"]]

di_final.head()

di_final_json_filename = "di_final.json"
di_final.to_json(di_final_json_filename, orient='records', indent=4)

import pandas as pd

dj = pd.read_json("hf://datasets/yigengx/PhySciBench/physcibench.json")

dj.head()

dj.isna().sum()

dj_final = dj.dropna(subset=["rubrics"])
dj_final.isna().sum()

dj_final = dj_final[['question', 'answer']]
dj_final.head()

dj_final_json_filename = "dj_final.json"
dj_final.to_json(dj_final_json_filename, orient="records", indent=4)

import pandas as pd

splits = {'train': 'ARC-Challenge/train-00000-of-00001.parquet', 'test': 'ARC-Challenge/test-00000-of-00001.parquet', 'validation': 'ARC-Challenge/validation-00000-of-00001.parquet'}
dm = pd.read_parquet("hf://datasets/allenai/ai2_arc/" + splits["train"])

dm.head()

dm.isna().sum()

dm_final = dm.drop(columns=["id"])
dm_final = dm_final.rename(columns={"answerKey" : "answer"})

dm_final.head()

dm_final_json_filename = "dm_final.json"
dm_final.to_json(dm_final_json_filename, orient="records", indent=4)

import pandas as pd

do = pd.read_json("hf://datasets/Abirate/english_quotes/quotes.jsonl", lines=True)

do.head()

do.isna().sum()

do_final = do[["quote", "author"]]
do_final = do_final.rename(columns={"quote" : "question", "author" : "answer"})

do_final_json_filename = "do_final.json"
do_final.to_json(do_final_json_filename, orient="records", indent=4)

from datasets import load_dataset

dp = load_dataset("BeIR/scifact", "corpus")

dp_corpus = dp["corpus"].to_pandas()

dp_corpus.head()

dp_corpus.isna().sum()

dp_final = dp_corpus.drop(columns=["_id"])
dp_final = dp_final.rename(columns={"title" : "question", "text" : "answer"})

dp_final.head()

dp_final_json_filename = "dp_final.json"
dp_final.to_json(dp_final_json_filename, orient="records", indent=4)

import pandas as pd

splits = {'train': 'train.csv', 'test': 'test.csv'}
dq = pd.read_csv("hf://datasets/okite97/news-data/" + splits["train"])

dq.head()

dq.isna().sum()

dq_final = dq[["Excerpt", "Category"]]
dq_final = dq_final.rename(columns={"Excerpt" : "question", "Category" : "answer"})

dq_final.head()

dq_final_json_filename = "dq_final.json"
dq_final.to_json(dq_final_json_filename, orient="records", indent=4)

import pandas as pd

splits = {'train': 'data/train-00000-of-00001-98aa5228a06a17d0.parquet', 'validation': 'data/validation-00000-of-00001-2553e47d408fab28.parquet', 'test': 'data/test-00000-of-00001-79fd931297fff765.parquet'}
du = pd.read_parquet("hf://datasets/climatebert/environmental_claims/" + splits["train"])

du.head()

du.isna().sum()

du_final = du.rename(columns={"text" : "question", "label" : "answer"})
du_final["answer"] = du_final["answer"].replace({1: "yes", 0: "no"})

du_final.head()

du_final_json_filename = "du_final.json"
du_final.to_json(du_final_json_filename, orient="records", indent=4)

import pandas as pd

dt = pd.read_csv("hf://datasets/zeroshot/arxiv-biology/abs-bio1.csv")

dt.head()

dt.isna().sum()

dt_final = dt[["title", "abstract"]]
dt_final = dt_final.rename(columns={"title" : "question", "abstract" : "answer"})

dt_final.head()

dt_final_json_filename = "dt_final.json"
dt_final.to_json(dt_final_json_filename, orient="records", indent=4)

import pandas as pd

dv = pd.read_csv("hf://datasets/grammarly/detexd-benchmark/test.csv")

dv.head()

dv.isna().sum()

dv_final = dv.drop(columns=["annotator_1", "annotator_2", "annotator_3"])
dv_final = dv_final.rename(columns={"text" : "question", "label" : "answer"})
dv_final["answer"] = dv_final["answer"].replace({1: "positive", 0: "negative"})

dv_final.head()

dv_final_json_filename = "dv_final.json"
dv_final.to_json(dv_final_json_filename, orient="records", indent=4)

import pandas as pd

splits = {'train': 'data/train-00000-of-00001-4b831beb8839bf3e.parquet', 'test': 'data/test-00000-of-00001-87f8706e009e9b75.parquet'}
dw = pd.read_parquet("hf://datasets/climatebert/climate_detection/" + splits["train"])

dw.head()

dw.isna().sum()

dw_final = dw.rename(columns={"text" : "question", "label" : "answer"})
dw_final["answer"] = dw_final["answer"].replace({1: "yes", 0: "no"})

dw_final.head()

dw_final_json_filename = "dw_final.json"
dw_final.to_json(dw_final_json_filename, orient="records", indent=4)

from datasets import load_dataset

dx = load_dataset("fraug-library/english_contractions_extensions", "contractions")

dx_train = dx["train"].to_pandas()

dx_train.head()

dx_train.isna().sum()

dx_final = dx_train.rename(columns={"contractions" : "question", "extensions" : "answer"})

dx_final.head()

dx_final_json_filename = "dx_final.json"
dx_final.to_json(dx_final_json_filename, orient="records", indent=4)

import pandas as pd

splits = {'train': 'train.csv', 'validation': 'validation.csv', 'test': 'test.csv'}
dy = pd.read_csv("hf://datasets/Myashka/SO-Python_QA-API_Usage-tanh_score/" + splits["train"])

dy.head()

dy.isna().sum()

dy_final = dy[["Question", "Answer"]]
dy_final = dy_final.rename(columns={"Question" : "question", "Answer" : "answer"})

dy_final.head()

dy_final_json_filename = "dy_final.json"
dy_final.to_json(dy_final_json_filename, orient="records", indent=4)

import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
dz = pd.read_json("hf://datasets/Amod/mental_health_counseling_conversations/combined_dataset.json", lines=True)

dz.head()

dz.isna().sum()

dz_final = dz.rename(columns={"Context" : "question", "Response" : "answer"})

dz_final.head()

dz_final_json_filename = "dz_final.json"
dz_final.to_json(dz_final_json_filename, orient="records", indent=4)

import pandas as pd

splits = {'train': 'data/Synthetic-Persona-Chat_train.csv', 'validation': 'data/Synthetic-Persona-Chat_valid.csv', 'test': 'data/Synthetic-Persona-Chat_test.csv'}
dat = pd.read_csv("hf://datasets/google/Synthetic-Persona-Chat/" + splits["train"])

dat.head()

dat.isna().sum()

dat_final = dat.rename(columns={"user 1 personas" : "question1", "user 2 personas" : "question2", "Best Generated Conversation" : "answer"})

dat_final.head()

dat_final_json_filename = "dat_final.json"
dat_final.to_json(dat_final_json_filename, orient="records", indent=4)

import pandas as pd

splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
das = pd.read_parquet("hf://datasets/ParlAI/blended_skill_talk/" + splits["train"])

das.head()

das.isna().sum()

das_final = das[["free_messages", "context"]]
das_final = das_final.rename(columns={"free_messages" : "question", "context" : "answer"})

das_final.head()

das_final_json_filename = "das_final.json"
das_final.to_json(das_final_json_filename, orient="records", indent=4)

import pandas as pd

splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
daa = pd.read_parquet("hf://datasets/Organika/wizard_of_wikipedia/" + splits["train"])

daa.head()

daa.isna().sum()

daa_final = daa.rename(columns={"persona" : "question", "text" : "answer"})

daa_final.head()

daa_final_json_filename = "daa_final.json"
daa_final.to_json(daa_final_json_filename, orient="records", indent=4)

import pandas as pd

splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet'}
dab = pd.read_parquet("hf://datasets/stanfordnlp/coqa/" + splits["train"])

dab.head()

dab.isna().sum()

dab_final = dab[["questions", "answers"]]
dab_final = dab_final.rename(columns={"questions" : "question", "answers" : "answer"})

dab_final.head()

dab_final_json_filename = "dab_final.json" 
dab_final.to_json(dab_final_json_filename, orient="records", indent=4)

import pandas as pd

splits = {'train': 'train.json', 'validation': 'valid.json', 'test': 'test.json'}
dac = pd.read_json("hf://datasets/allenai/prosocial-dialog/" + splits["train"], lines=True)

dac.head()

dac.isna().sum()

dac_final = dac[["context", "response"]]
dac_final = dac_final.rename(columns={"context" : "question", "response" : "answer"})

dac_final.head()

dac_final_json_filename = "dac_final.json"
dac_final.to_json(dac_final_json_filename, orient="records", indent=4)

import pandas as pd

splits = {'train': 'train.parquet', 'validation': 'valid.parquet', 'test': 'test.parquet'}
dad = pd.read_parquet("hf://datasets/allenai/soda/" + splits["train"])

dad.head()

dad.isna().sum()

dad_final = dad[["narrative", "literal"]]
dad_final = dad_final.rename(columns={"narrative" : "question", "literal" : "answer"}) 

dad_final.head()

dad_final_json_filename = "dad_final.json"
dad_final.to_json(dad_final_json_filename, orient="records", indent=4)

import pandas as pd

splits = {'train': 'data/train-00000-of-00001-b42a775f407cee45.parquet', 'validation': 'data/validation-00000-of-00001-134b8fd0c89408b6.parquet'}
dae = pd.read_parquet("hf://datasets/OpenAssistant/oasst1/" + splits["train"])

dae.head()

dae.isna().sum()

dae_final = dae[["text", "role"]]
dae_final = dae_final.rename(columns={"text" : "question", "role" : "answer"})

dae_final.head()

dae_final_json_filename = "dae_final.json"
dae_final.to_json(dae_final_json_filename, orient="records", indent=4)

import pandas as pd

daf = pd.read_json("hf://datasets/databricks/databricks-dolly-15k/databricks-dolly-15k.jsonl", lines=True)

daf.head()

daf.isna().sum()

daf_final = daf[["instruction", "response"]]
daf_final = daf_final.rename(columns={"instruction" : "question", "response" : "answer"})

daf_final.head()

daf_final_json_filename = "daf_final.json"
daf_final.to_json(daf_final_json_filename, orient="records", indent=4)

from datasets import load_dataset

dag = load_dataset("sentence-transformers/codesearchnet")

dag_train = dag["train"].to_pandas()
dag_train.head()

dag_train.isna().sum()

dag_final = dag_train.rename(columns={"code" : "answer", "comment" : "question"})

dag_final.head()

dag_final_json_filename = "dag_final.json"
dag_final.to_json(dag_final_json_filename, orient="records", indent=4)

import pandas as pd

dah = pd.read_parquet("hf://datasets/iamtarun/code_instructions_120k_alpaca/data/train-00000-of-00001-d9b93805488c263e.parquet")

dah.head()

dah.isna().sum()

dah_final = dah[["instruction", "output"]]
dah_final = dah_final.rename(columns={"instruction" : "question", "output" : "answer"})

dah_final.head()

dah_final_json_filename = "dah_final.json"
dah_final.to_json(dah_final_json_filename, orient="records", indent=4)

import pandas as pd

splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
dai = pd.read_parquet("hf://datasets/allenai/sciq/" + splits["train"])

dai.head()

dai.isna().sum()

dai_final = dai[["question", "correct_answer"]]
dai_final = dai_final.rename(columns={"correct_answer" : "answer"})

dai_final.head()

dai_final_json_filename = "dai_final.json"
dai_final.to_json(dai_final_json_filename, orient="records", indent=4)

import pandas as pd

splits = {'train': 'additional/train-00000-of-00001.parquet', 'validation': 'additional/validation-00000-of-00001.parquet', 'test': 'additional/test-00000-of-00001.parquet'}
daj = pd.read_parquet("hf://datasets/allenai/openbookqa/" + splits["train"])

daj.head()

daj.isna().sum()

daj_final = daj[["question_stem", "choices", "answerKey"]]
daj_final = daj_final.rename(columns={"question_stem" : "question", "answerKey" : "answer"})

daj_final.head()

daj_final_json_filename = "daj_final.json"
daj_final.to_json(daj_final_json_filename, orient="records", indent=4)

import pandas as pd

splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet'}
dak = pd.read_parquet("hf://datasets/allenai/qasc/" + splits["train"])

dak.head()

dak.isna().sum()

dak_final = dak[["question", "choices", "answerKey"]]
dak_final = dak_final.rename(columns={"question" : "question", "answerKey" : "answer"})

dak_final.head()

dak_final_json_filename = "dak_final.json"
dak_final.to_json(dak_final_json_filename, orient="records", indent=4)

import pandas as pd

dal = pd.read_parquet("hf://datasets/qiaojin/PubMedQA/pqa_artificial/train-00000-of-00001.parquet")

dal.head()

dal.isna().sum()

dal_final = dal[["question", "long_answer"]]
dal_final = dal_final.rename(columns={"question" : "question", "long_answer" : "answer"})

dal_final.head()

dal_final_json_filename = "dal_final.json"
dal_final.to_json(dal_final_json_filename, orient="records", indent=4)

import pandas as pd

dam = pd.read_json("hf://datasets/billli/QuRe/QuRe.json", lines=True)

dam.head()

dam.isna().sum()

dam_final = dam[["orig_sentence", "quant_sent"]]
dam_final = dam_final.rename(columns={"orig_sentence" : "question", "quant_sent" : "answer"})

dam_final.head()

dam_final_json_filename = "dam_final.json"
dam_final.to_json(dam_final_json_filename, orient="records", indent=4)

from datasets import load_dataset

dan = load_dataset("hotpotqa/hotpot_qa", "distractor")

dan_train = dan["train"].to_pandas()

dan_train.head()

dan_train.isna().sum()

dan_final = dan_train[["question", "answer"]]
dan_final.head()

dan_final_json_filename = "dan_final.json"
dan_final.to_json(dan_final_json_filename, orient="records", indent=4)

import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
splits = {'train': 'squad_v2/train-00000-of-00001.parquet', 'validation': 'squad_v2/validation-00000-of-00001.parquet'}
dao = pd.read_parquet("hf://datasets/rajpurkar/squad_v2/" + splits["train"])

dao.head()

dao.isna().sum()

dao_final = dao[["question", "answers"]]
dao_final = dao_final.rename(columns={"question" : "question", "answers" : "answer"})

dao_final.head()

dao_final_json_filename = "dao_final.json"
dao_final.to_json(dao_final_json_filename, orient="records", indent=4)

import pandas as pd

splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet'}
dap = pd.read_parquet("hf://datasets/google/boolq/" + splits["train"])

dap.head()

dap.isna().sum()

dap_final = dap[["question", "answer"]]
dap_final.head()

dap_final_json_filename = "dap_final.json"
dap_final.to_json(dap_final_json_filename, orient="records", indent=4)

import pandas as pd

splits = {'test': 'data/test-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet', 'train': 'data/train-00000-of-00001.parquet'}
daq = pd.read_parquet("hf://datasets/microsoft/wiki_qa/" + splits["test"])

daq.head()

daq.isna().sum()

daq_final = daq[["question", "answer"]]
daq_final["question"] = daq_final["question"].str.lower()

daq_final.head()

daq_final_json_filename = "daq_final.json"
daq_final.to_json(daq_final_json_filename, orient="records", indent=4)

import pandas as pd

splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
dar = pd.read_parquet("hf://datasets/tau/commonsense_qa/" + splits["train"])

dar.head()

dar.isna().sum()

dar_final = dar[["question", "choices", "answerKey"]]
dar_final = dar_final.rename(columns={"answerKey" : "answer"})

dar_final.head()

dar_final_json_filename = "dar_final.json"
dar_final.to_json(dar_final_json_filename, orient="records", indent=4)

import pandas as pd

splits = {'train': 'data/train-00000-of-00001-506370352f622815.parquet', 'test': 'data/test-00000-of-00001-bae602f3ee37f4ca.parquet'}
dau = pd.read_parquet("hf://datasets/ChilleD/StrategyQA/" + splits["train"])

dau.head()

dau.isna().sum()

dau_final = dau[["question", "answer"]]
dau_final.head()

dau_final_json_filename = "dau_final.json"
dau_final.to_json(dau_final_json_filename, orient="records", indent=4)

import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
dav = pd.read_parquet("hf://datasets/EleutherAI/race/high/test-00000-of-00001.parquet")

dav.head()

dav.isna().sum()

dav_final = dav.rename(columns={"article" : "question", "problems" : "answer"})

dav_final.head()

dav_final_json_filename = "dav_final.json"
dav_final.to_json(dav_final_json_filename, orient="records", indent=4)

import pandas as pd

daw = pd.read_parquet("hf://datasets/qwedsacf/competition_math/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet")

daw.head()

daw.isna().sum()

daw_final = daw[["problem", "solution"]]
daw_final = daw_final.rename(columns={"problem" : "question", "solution" : "answer"})

daw_final.head()

daw_final_json_filename = "daw_final.json"
daw_final.to_json(daw_final_json_filename, orient="records", indent=4)

import pandas as pd

splits = {'test': 'abstract_algebra/test-00000-of-00001.parquet', 'validation': 'abstract_algebra/validation-00000-of-00001.parquet', 'dev': 'abstract_algebra/dev-00000-of-00001.parquet'}
dax = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])

dax.head()

dax.isna().sum()

dax_final = dax[["question", "choices", "answer"]]
dax_final.head()

dax_final_json_filename = "dax_final.json"
dax_final.to_json(dax_final_json_filename, orient="records", indent=4)

import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
day = pd.read_json("hf://datasets/agentlans/first-person-dialogue/first-person-dialogue_k1000.jsonl.zst", lines=True)

day.head()

day.isna().sum()

day_final = day.rename(columns={"conversations" : "question", "source" : "answer"})

day_final.head()

day_final_json_filename = "day_final.json"
day_final.to_json(day_final_json_filename, orient="records", indent=4)

import pandas as pd

daz = pd.read_json("hf://datasets/jimmyzxj/drosophila-literature-corpus/corpus.jsonl", lines=True)

daz.head()

daz.isna().sum()

daz_final = daz[["title", "abstract"]]
daz_final = daz_final.rename(columns={"title" : "question", "abstract" : "answer"})

daz_final.head()

daz_final_json_filename = "daz_final.json"
daz_final.to_json(daz_final_json_filename, orient="records", indent=4)

import pandas as pd

dba = pd.read_json("hf://datasets/schneewolflabs/hecke-dpo/data/train.jsonl", lines=True)
dba.head()

dba.isna().sum()

dba_final = dba[["prompt", "chosen"]]
dba_final = dba_final.rename(columns={"prompt" : "question", "chosen" : "answer"})

dba_final.head()

dba_final_json_filename = "dba_final.json"
dba_final.to_json(dba_final_json_filename, orient="records", indent=4)

import pandas as pd

dbb = pd.read_json("hf://datasets/ianncity/GLM-5.2-Conversation/dataset.jsonl", lines=True)

dbb.head()

dbb.isna().sum()

dbb_final = dbb.rename(columns={"messages" : "text"})

dbb_final.head()

dbb_final_json_filename = "dbb_final.json"
dbb_final.to_json(dbb_final_json_filename, orient="records", indent=4)

import pandas as pd

dbc = pd.read_json("hf://datasets/MuskumPillerum/General-Knowledge/output.json")

dbc.head()

dbc.isna().sum()

dbc_final = dbc.dropna(subset=["Answer"])
dbc_final = dbc_final.rename(columns={"Question" : "question", "Answer" : "answer"})

dbc_final.head()

dbc_final_json_filename = "dbc_final.json"
dbc_final.to_json(dbc_final_json_filename, orient="records", indent=4)

from datasets import load_dataset

dbd = load_dataset("ExponentialScience/DLT-Scientific-Literature")

dbd_train = dbd["train"].to_pandas()

dbd_train.head()

dbd_train.isna().sum()

dbd_final = dbd_train.dropna(subset=["abstract", "year", "publicationDate"])
dbd_final = dbd_final["title"]
dbd_final = pd.DataFrame(dbd_final)
dbd_final = dbd_final.rename(columns={"title" : "question"})

dbd_final.head()

dbd_final_json_filename = "dbd_final.json"
dbd_final.to_json(dbd_final_json_filename, orient="records", indent=4)

import pandas as pd

dbe = pd.read_json("hf://datasets/ianncity/GLM-5.2-Science/dataset.jsonl", lines=True)

dbe.head()

dbe.isna().sum()

dbe_final = dbe.rename(columns={"messages" : "text"})

dbe_final.head()

dbe_final_json_filename = "dbe_final.json"
dbe_final.to_json(dbe_final_json_filename, orient="records", indent=4)

import pandas as pd

splits = {'train': 'train.jsonl', 'validation': 'validation.jsonl'}
dbf = pd.read_json("hf://datasets/grammarly/coedit/" + splits["train"], lines=True)

dbf.head()

dbf.isna().sum()

dbf_final = dbf[["tgt", "src"]]
dbf_final = dbf_final.rename(columns={"tgt" : "question", "src" : "answer"})

dbf_final.head()

dbf_final_json_filename = "dbf_final.json"
dbf_final.to_json(dbf_final_json_filename, orient="records", indent=4)

import pandas as pd

dbg = pd.read_json("hf://datasets/ianncity/GLM-5.2-Finance-80000x/dataset.jsonl", lines=True)

dbg.head()

dbg.isna().sum()

dbg_final = dbg[["transcript"]]
dbg_final = dbg_final.rename(columns={"transcript" : "text"})

dbg_final.head()

dbg_final_json_filename = "dbg_final.json"
dbg_final.to_json(dbg_final_json_filename, orient="records", indent=4)

from datasets import load_dataset

dbh = load_dataset("Isotonic/human_assistant_conversation_deduped")

dbh_train = dbh["train"].to_pandas()

dbh_train.head()

dbh_train.isna().sum()

dbh_final = dbh_train[["text"]]

dbh_final.head()

dbh_final_json_filename = "dbh_final.json"
dbh_final.to_json(dbh_final_json_filename, orient="records", indent=4)

from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
dbj = load_dataset("Krisl7286/HH-RLHF-Personal-Copy-Helpfulness-First")

dbj_train = dbj["train"].to_pandas()

dbj_train.head()

dbj_train.isna().sum()

dbj_final = dbj_train[["chosen"]]
dbj_final = dbj_final.rename(columns={"chosen" : "text"})

dbj_final.head()

dbj_final_json_filename = "dbj_final.json"
dbj_final.to_json(dbj_final_json_filename, orient="records", indent=4)

import pandas as pd

dbk = pd.read_parquet("hf://datasets/alon-albalak/template-test-results-first-person/data/train-00000-of-00001.parquet")

dbk.head()

dbk.isna().sum()

dbk_final = dbk[["creative_preps", "responses"]]
dbk_final = dbk_final.rename(columns={"creative_preps" : "question", "responses" : "answer"})

dbk_final.head()

dbk_final_json_filename = "dbk_final.json"
dbk_final.to_json(dbk_final_json_filename, orient="records", indent=4)

import pandas as pd

dbl = pd.read_parquet("hf://datasets/alon-albalak/template-test-results-first-person-v2/data/train-00000-of-00001.parquet")

dbl.head()

dbl.isna().sum()

dbl_final = dbl[["creative_preps", "responses"]]
dbl_final = dbl_final.rename(columns={"creative_preps" : "question", "responses" : "answer"})

dbl_final.head()

dbl_final_json_filename = "dbl_final.json"
dbl_final.to_json(dbl_final_json_filename, orient="records", indent=4)

import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
dbm = pd.read_json("hf://datasets/TajaKuzmanPungersek/X-GENRE-text-genre-dataset/X-GENRE-train.jsonl", lines=True)

dbm.head()

dbm.isna().sum()

dbm_final = dbm[["text"]]

dbm_final.head()

dbm_final_json_filename = "dbm_final.json"
dbm_final.to_json(dbm_final_json_filename, orient="records", indent=4)

import pandas as pd

splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
dbn = pd.read_parquet("hf://datasets/HuggingFaceH4/no_robots/" + splits["train"])

dbn.head()

dbn.isna().sum()

dbn_final = dbn[["prompt", "messages"]]
dbn_final = dbn_final.rename(columns={"prompt" : "question", "messages" : "answer"})

dbn_final.head()

dbn_final_json_filename = "dbn_final.json"
dbn_final.to_json(dbn_final_json_filename, orient="records", indent=4)

import pandas as pd

dbo = pd.read_parquet("hf://datasets/openai/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet")

dbo.head()

dbo.isna().sum()

dbo_final = dbo[["prompt", "canonical_solution"]]
dbo_final = dbo_final.rename(columns={"prompt" : "question", "canonical_solution" : "answer"})

dbo_final.head()

dbo_final_json_filename = "dbo_final.json"
dbo_final.to_json(dbo_final_json_filename, orient="records", indent=4)


# Login using e.g. `huggingface-cli login` to access this dataset
dbp = load_dataset("PrimeIntellect/SYNTHETIC-2-SFT-verified")

dbp_train = dbp["train"].to_pandas()

dbp_train.head()

dbp_train.isna().sum()

dbp_final = dbp_train[["messages"]]
dbp_final = dbp_final.rename(columns={"messages" : "text"})

dbp_final.head()

dbp_final_json_filename = "dbp_final.json"
dbp_final.to_json(dbp_final_json_filename, orient="records", indent=4)

import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet'}
dbq = pd.read_parquet("hf://datasets/ethicalabs/Kurtis-E1-SFT/" + splits["train"])

dbq.head()

dbq.isna().sum()

dbq_final = dbq[["question", "answer"]]

dbq_final.head()

dbq_final_json_filename = "dbq_final.json"
dbq_final.to_json(dbq_final_json_filename, orient="records", indent=4)

from datasets import load_dataset

dbr = load_dataset("BEE-spoke-data/wikipedia-20230901.en-deduped", "text-only")

dbr_train = dbr["train"].to_pandas()
dbr_train.head()
dbr_train.isna().sum()

dbr_final = dbr_train.dropna(subset=["text"]).copy()
dbr_final = dbr_final[["text"]]

dbr_final.head()

dbr_final_json_filename = "dbr_final.json"
dbr_final.to_json(dbr_final_json_filename, orient="records", indent=4)

from datasets import load_dataset

dbt = load_dataset("EliMC/TxT360-5M-sample-en", "default")

dbt_train = dbt["train"].to_pandas()
dbt_train.head()
dbt_train.isna().sum()

dbt_final = dbt_train.dropna(subset=["text"]).copy()
dbt_final = dbt_final[["text"]]

dbt_final.head()

dbt_final_json_filename = "dbt_final.json"
dbt_final.to_json(dbt_final_json_filename, orient="records", indent=4)
