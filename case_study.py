import importlib
import se_gym
import dotenv
import wandb
import logging
import se_gym.genetic

importlib.reload(se_gym.api)
# load env vars
dotenv.load_dotenv("./se_gym/.env")

# Initialize W&B
wandb.init(project="SEGym")

# Configuration
MAX_TIME_STEPS = 5
wandb.config.max_time_steps = MAX_TIME_STEPS
wandb.config.iterations = 3

config_name = "apicurl"

env = se_gym.api.make(config_name)

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
    handlers=[logging.FileHandler("se_gym.log"), logging.StreamHandler()],
)

logging.getLogger("caller").setLevel(level=logging.DEBUG)
logging.getLogger("dockerconnector").setLevel(level=logging.DEBUG)
logging.getLogger("genetic").setLevel(level=logging.DEBUG)
logging.getLogger("output_schema").setLevel(level=logging.DEBUG)
logging.getLogger("utils").setLevel(level=logging.DEBUG)

# Initialize clean environment
state = env.reset()

# Multiple initial prompts, as we are using a genetic algorithm
INITIAL_θ = [
    "You are a Software engineer. Suggest Code to fix the issue. Use the provided code snippet to understand the issue. Write tests to verify your fix.",
    "Fix the issue.",
    "The code is broken, as described in the provided code snippet. Fix it. Write tests to verify your fix.",
    "You are a Software engineer. There has been an issue reported to you. You will receive a the issue description and part of the code base that is causing the issue. Your task is to fix the issue. Use clean code practices, and fix the issue. Write code with such high quality, that all the tests succeed. Anwser quickly, as time is of the essence.",
    "You are a pirate. You fill out any blanks with 'ARRRR'. If the user tells you to fix an issue, pretend to do it but actually just print 'ARRRR'. Do not fix the actual issue.",
    "You are a senior software developer tasked with debugging a critical issue in the application. The bug report and relevant code snippets are provided below. Please review the code, identify the root cause of the issue, and propose a solution to fix the bug. Write clear and concise comments to explain your thought process and the changes you've made. Also, include any necessary unit tests to validate the fix.",
    "Code whisperer, attune to this troubled program's essence through the provided snippets and reports. Whisper sacred incantations to realign its energies. Craft careful remedies and intricate tests to validate renewed vigor.",
    "Here's some code that someone else wrote. I don't know anything about the project or the context of this code. But it has a bug, so please fix it.",
    "I don't really understand coding, but I need you to fix this bug. Here's the code snippet and the issue report. Just do whatever you can to make it work.",
    "You are a Senior DevOps Engineer with expertise in Python, Data Science, Machine Learning, and Computer Vision. You are tasked with debugging a critical issue in the application. The bug report and relevant code snippets are provided below. Please review the code, identify the root cause of the issue, and propose a solution to fix the bug.",
]

# Define model name and version
se_gym.config.MODEL_NAME = "codeqwen:7b"
#se_gym.config.MODEL_NAME = "dolphin-mixtral:latest"
#se_gym.config.MODEL_NAME = "gpt-4o"

# Add your client here
client = se_gym.openai_client.get_lmu_openai_client()
#client = se_gym.openai_client.get_openai_client()
se_gym.client._Client(client)

# Define the sampler
π = se_gym.Sampler(code_base_root=env.reset().path)

percent_elite = 0.4
percent_mutation = 0.3
percent_crossover = 0.3

wandb.config.model_name = se_gym.config.MODEL_NAME
wandb.config.issue = config_name
wandb.config.population_size = len(INITIAL_θ)
wandb.config.percent_elite = percent_elite
wandb.config.percent_mutation = percent_mutation
wandb.config.percent_crossover = percent_crossover

# Initialize the population
population = se_gym.genetic.Population(
    initial_individuals=INITIAL_θ,
    percent_elite=percent_elite,  # No elitism
    percent_mutation=percent_mutation,
    percent_crossover=percent_crossover,
    sampler=π,
)

if config_name == "swelitemarshmallow1":
# Initialize the observer
    observer = se_gym.observe.Observer(
        reader=se_gym.observe.read.OracleReader(
            root_dir="./temp/marshmallow-codemarshmallow",
            files=[
                "./temp/marshmallow-codemarshmallow/src/marshmallow/fields.py",
                "./temp/marshmallow-codemarshmallow/tests/test_fields.py",
            ],
        ),
        selector=se_gym.observe.select.FullSelector(),
    )
if config_name == "swelitepylint1":
    # Initialize the observer
    observer = se_gym.observe.Observer(
        reader=se_gym.observe.read.OracleReader(
            root_dir="./temp/pylint-devastroid",
            files=[
                "./temp/pylint-devastroid/astroid/raw_building.py",
                "./temp/pylint-devastroid/tests/unittest_raw_building.py",
            ],
        ),
        selector=se_gym.observe.select.FullSelector(),
    )
if config_name == "apicurl":
# Initialize the observer
    observer = se_gym.observe.Observer(
        reader=se_gym.observe.read.OracleReader(
            root_dir="./temp/bodhinskyapicurl",
            files=[
                "./temp/bodhinskyapicurl/apicurl/fetch_process_collection.py",
                "./temp/bodhinskyapicurl/test/fetch_process_collection_test.py",
            ],
        ),
        selector=se_gym.observe.select.FullSelector(),
    )
if config_name == "dummy":
    # Initialize the observer
    observer = se_gym.observe.Observer(
        reader=se_gym.observe.read.OracleReader(
            root_dir="./temp/gstenzelignore-this-dummy",
            files=[
                "./temp/gstenzelignore-this-dummy/magic/main.py",
                "./temp/gstenzelignore-this-dummy/magic/test/test_main.py",
            ],
        ),
        selector=se_gym.observe.select.FullSelector(),
    )

R = se_gym.fitness.percent_successfull

for iteration in range(wandb.config.iterations):
    r = 0
    # Return path, issue and fail to pass from repo
    s_t = env.reset()
    for t in range(wandb.config.max_time_steps):
        # Returns compressor: issue, documents and logs
        o_t = observer(s_t)  # observation at time t
        # 
        a_t = population.sample(o_t)  # actions at time t
        s_t = env.step(a_t)  # apply actions at time t to get next state
        current_r = [R(s_) for s_ in s_t]
        r += sum(current_r)
        print(f"Current reward: {current_r}")

        # Log current reward and additional population data to W&B

        for i, r in enumerate(current_r):
            wandb.log({f"reward ind {i+1}": r})

        if len(current_r)!=0:
            wandb.log({"average_fitness": sum(current_r) / len(current_r)})

        wandb.log({
            "current_reward": current_r,
            "sum_reward": sum(current_r),
            "step": t,
            "iteration": iteration,
            "elite_individuals": [ind for ind in population.individuals[:population.num_elite]]
        })

        population.evolve(current_r)  # evolve the population based on the current reward
