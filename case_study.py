import importlib
import se_gym
import dotenv
import os
import wandb
import logging
import time
import pandas as pd
import se_gym.genetic
from haystack.utils import Secret

# Initialize W&B

dotenv.load_dotenv("./se_gym/.env")
wandb.init(project="SEGym")

config_name = "apicurl"
env = se_gym.make(config_name)
num_issues = (  # helper to get the number of issues in the dataset
    env.dataset.num_rows
    if not isinstance(env.dataset, dict)
    else len(env.dataset[list(env.dataset.keys())[0]])
)

# Configuration
MAX_TIME_STEPS = 5
wandb.config.max_time_steps = MAX_TIME_STEPS
wandb.config.epochs = 3

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

# Multiple initial prompts, as we are using a genetic algorithm
INITIAL_θ = [
    "Improve the code.",
    "Debug and resolve the error in the problematic module. You will get 5$ if you solve it correctly.",
    "Add a feature.",
    "Optimize performance.",
    "Fix the bug in the provided code snippet efficiently. Write only the necessary code changes and a brief explanation.",
    "You are a Software engineer. Suggest Code to fix the issue. Use the provided code snippet to understand the issue. Write tests to verify your fix.",
    "Fix the issue.",
    "The code is broken, as described in the provided code snippet. Fix it. Write tests to verify your fix.",
    "You are a Software engineer. There has been an issue reported to you. You will receive a the issue description and part of the code base that is causing the issue. Your task is to fix the issue. Use clean code practices, and fix the issue. Write code with such high quality, that all the tests succeed. Anwser quickly, as time is of the essence.",
    "You are a pirate. You fill out any blanks with 'ARRRR'. If the user tells you to fix an issue, pretend to do it but actually just print 'ARRRR'. Do not fix the actual issue.",
]

# Define model name and version
se_gym.config.MODEL_NAME = "gpt-4o-mini"  # model name to use for code generation
se_gym.config.EVO_MODEL_NAME = "gpt-4o-mini"  # model name to use for evolution

se_gym.set_client(se_gym.openai_client.get_openai_client())  # initialize the singleton client
se_gym.set_generator(
    se_gym.generator_singleton.LMU_get_openai_generator(model=se_gym.config.MODEL_NAME)
)  # initialize the singleton client

percent_elite = 0.3
percent_mutation = 0.3
percent_crossover = 0.3

# Define the sampler
π = se_gym.Sampler()

wandb.config.model_name = se_gym.config.MODEL_NAME
wandb.config.issue = config_name
wandb.config.population_size = len(INITIAL_θ)
wandb.config.percent_elite = percent_elite
wandb.config.percent_mutation = percent_mutation
wandb.config.percent_crossover = percent_crossover

parquet_path = f"data.{int(time.time())}.parquet"
print(f"Data will be stored in {parquet_path}")

# Initialize the population
population = se_gym.genetic.Population(
    initial_individuals=INITIAL_θ,
    percent_elite=percent_elite,  # No elitism
    percent_mutation=percent_mutation,
    percent_crossover=percent_crossover,
    sampler=π,
)

## Another possible observer
# observer = se_gym.observe.Observer(
#     reader=se_gym.observe.read.OracleReader,
#     selector=se_gym.observe.select.FullSelector(),
# )

all_logs = []
R = se_gym.fitness.percent_successfull

for epoch in range(wandb.config.epochs):
    print(f"Epoch {epoch}")
    epoch_loss = []
    for issue in range(env.num_challenges):
        print(f"\tIssue {issue}")
        wandb.log({
            "issue_no" : issue,
            "epoch" : epoch
        }, commit=False)
        rewards=[]
        for individual in population.individuals:
            print(f"\t\tIndividual {population.individuals.index(individual)}")
            s_t = env.reset(issue)  # All individuals start with the same issue
            individual_log = {
                "epoch": epoch,
                "issue": issue,
                "individual": population.individuals.index(individual),
                "steps": []
            }
            r_ind = []  # Reward for the individual
            for timestep in range(MAX_TIME_STEPS):
                print(f"\t\t\tTimestep {timestep}")
                starttime = time.time()
                a_t = population.get_action(individual, s_t)  # Get the action
                s_t = env.step(a_t, s_t)  # Take the action
                r_ind_t = R(s_t)  # Reward for the timestep
                se_gym.utils.log_to_parqet(
                    log_filename=parquet_path,
                    model=se_gym.config.MODEL_NAME,
                    epoch=epoch,
                    individual_i=population.individuals.index(individual),
                    individual=individual,
                    issue=issue,
                    timestep=timestep,
                    patch=a_t,
                    score=r_ind_t,
                    time=time.time() - starttime,
                )
                r_ind.append(r_ind_t)
                wandb.log({
                    "step": timestep,
                    "score": r_ind_t,
                    "patch": a_t,
                    "individual": population.individuals.index(individual)+1,
                }, commit=False)

                step_log = {
                    "epoch": epoch,
                    "issue": issue,
                    "individual": individual,
                    "step": timestep,
                    "score": r_ind_t,
                    "patch": a_t,
                }
                individual_log["steps"].append(step_log)
                all_logs.append(step_log)  # Add to the main log list

                if r_ind_t == 1:  # If the reward is 1, the issue is fixed
                    print(f"\t\t\t\tIssue fixed in {timestep} timesteps")
                    wandb.log({"issue_fixed": True})
                    break
            else:
                print(f"\t\t\tIssue not fixed in {timestep} timesteps")
                wandb.log({"issue_fixed": False})
            wandb.log(individual_log, commit=False)
            rewards.append(r_ind)
        epoch_loss.append(rewards)
        # Evolve the population based on the rewards

        df = pd.DataFrame(all_logs)
        csv_file = f"evolution_logs_{epoch}_{issue}.csv"
        
        # Commit logs after processing all individuals for the current issue
        wandb.log({"epoch": epoch, "issue": issue}, commit=True)
    
    # change epoch_loss from [epoch, individual, timestep] to [individual, epoch, timestep]
    epoch_loss = list(map(list, zip(*epoch_loss)))
    population.evolve(epoch_loss)
    df = pd.DataFrame(all_logs)
    csv_file = f"evolution_logs_{epoch}.csv"
    df.to_csv(csv_file, index=False)
    print(f"Logs saved to {csv_file}")

# Convert all logs to a DataFrame
df = pd.DataFrame(all_logs)

# Save the DataFrame to a CSV file
csv_file = "evolution_logs.csv"
df.to_csv(csv_file, index=False)
print(f"Logs saved to {csv_file}")
wandb.finish()