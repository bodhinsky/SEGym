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
wandb.init(project="dummy")

# Configuration
MAX_TIME_STEPS = 20
wandb.config.max_time_steps = MAX_TIME_STEPS
wandb.config.iterations = 5

env = se_gym.api.make("apicurl")

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
    "Ancient code monk, meditate on this program's snippets and reports to unravel its tangled logic. Craft brushstroke solutions with precision. Guide future generations through enlightening tests."
]

# Define model name and version
#se_gym.config.MODEL_NAME = "starcoder2:instruct"
#se_gym.config.MODEL_NAME = "llama3:8b"
se_gym.config.MODEL_NAME = "gpt-4o"
wandb.config.model_name = se_gym.config.MODEL_NAME

# Add your client here
#client = se_gym.openai_client.get_lmu_openai_client()
client = se_gym.openai_client.get_openai_client()
se_gym.client._Client(client)

# Define the sampler
π = se_gym.Sampler(code_base_root=env.reset().path)

percent_elite = 0.2
percent_mutation = 0.6
percent_crossover = 0.2

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

# Initialize the observer
observer = se_gym.observe.Observer(
    reader=se_gym.observe.read.OracleReader(
        root_dir="./temp/bodhinskyapicurl",
        files=[
            "./temp/bodhinskyapicurl/src/apiCurl/fetchProcessCollection.py",
            "./temp/bodhinskyapicurl/test/artistOverview_test.py",
            "./temp/bodhinskyapicurl/test/fetchProcessCollection_test.py",
            "./temp/bodhinskyapicurl/src/apiCurl/__main__.py",
            "./temp/bodhinskyapicurl/src/apiCurl/userAuth.py",
        ],
    ),
    selector=se_gym.observe.select.FullSelector(),
)

R = se_gym.fitness.num_failed_tests

for iteration in range(wandb.config.iterations):
    r = 0
    s_t = env.reset()
    for t in range(wandb.config.max_time_steps):
        o_t = observer(s_t)  # observation at time t
        a_t = population.sample(o_t)  # actions at time t
        s_t = env.step(a_t)  # apply actions at time t to get next state
        current_r = [R(s_) for s_ in s_t]
        r += sum(current_r)
        print(f"Current reward: {current_r}")

        # Log current reward and additional population data to W&B
        wandb.log({
            "reward": sum(current_r),
            "population_size": len(population.individuals),
            "iteration": iteration,
            "average_fitness": sum(current_r) / len(current_r),
            "elite_individuals": [ind for ind in population.individuals[:population.num_elite]]
        })

        population.evolve(current_r)  # evolve the population based on the current reward
