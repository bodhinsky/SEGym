#!/bin/bash

# Default values
CONFIG_FILE="config.env"
MODEL_FLAG="gpt-4o-mini"
API_ENDPOINT="openai"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -c|--config)
      CONFIG_FILE="$2"
      shift # past argument
      shift # past value
      ;;
    -m|--model)
      MODEL_FLAG="$2"
      shift # past argument
      shift # past value
      ;;
    -a|--api)
      API_ENDPOINT="$2"
      shift # past argument
      shift # past value
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

# Validate required flags
if [[ -z "$CONFIG_FILE" ]]; then
  echo "No config file specified. Please provide a config file with -c or --config."
  exit 1
fi

if [[ -z "$MODEL_FLAG" ]]; then
  echo "No model specified. Please provide a model flag with -m or --model."
  exit 1
fi

if [[ -z "$API_ENDPOINT" ]]; then
  echo "No API endpoint specified. Please provide an API endpoint with -a or --api."
  echo "Options: openai, lmu-ollama"
  exit 1
fi

# Load the specified configuration file
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Config file '$CONFIG_FILE' does not exist!"
  exit 1
fi

# Load configuration from the config.env file
source ./config.env

# Run the Python script with the specified parameters
python3 <<EOF
import os
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

import se_gym
import wandb
import time
import pandas as pd
import se_gym.genetic

# Initialize the environment
config_name = "${CONFIG_NAME}"
env = se_gym.make(config_name)

wandb.require("core")
wandb.init(project="${WANDB_PROJECT}")

# Set up the number of issues
num_issues = (
    env.dataset.num_rows
    if not isinstance(env.dataset, dict)
    else len(env.dataset[list(env.dataset.keys())[0]])
)

# Configuration
MAX_TIME_STEPS = int(os.getenv("MAX_TIME_STEPS", "${MAX_TIME_STEPS}"))
wandb.config.max_time_steps = MAX_TIME_STEPS
wandb.config.epochs = int(os.getenv("EPOCHS", "${EPOCHS}"))

# Set the API key based on the provided endpoint flag
if "${API_ENDPOINT}" == "openai":
    se_gym.config.MODEL_CONFIG = se_gym.config.EVO_MODEL_CONFIG = (
        se_gym.config.RETRIEVER_MODEL_CONFIG
    ) = dict(
        model_name="${MODEL_FLAG}",
    )

    se_gym.config.MODEL_CONFIG["api_key"] = os.getenv("OPENAI_API_KEY")
    se_gym.config.EVO_MODEL_CONFIG["api_key"] = os.getenv("OPENAI_API_KEY")
    se_gym.config.RETRIEVER_MODEL_CONFIG["api_key"] = os.getenv("OPENAI_API_KEY")
elif "${API_ENDPOINT}" == "lmu-ollama":
    se_gym.config.LLAMACPP_COMPATIBLE_SCHEMA = True

    se_gym.config.MODEL_CONFIG = se_gym.config.EVO_MODEL_CONFIG = (
        se_gym.config.RETRIEVER_MODEL_CONFIG
    ) = dict(
        base_url="https://ollama.mobile.ifi.lmu.de/v1/",
        # base_url="http://10.153.199.193:11434/v1/",
        # base_url="http://10.153.199.193:1234/v1/",
        api_key="ollama",
        model_name="${MODEL_FLAG}",
    )

    se_gym.generators.patch_openai_auth() # Patch OpenAI gym to use BasicAuth using a .env file

se_gym.utils.logging_setup()

# Multiple initial prompts, as we are using a genetic algorithm
INITIAL_θ = ${INITIAL_THETA}

percent_elite = float(os.getenv("PERCENT_ELITE", "${PERCENT_ELITE}"))
percent_mutation = float(os.getenv("PERCENT_MUTATION", "${PERCENT_MUTATION}"))
percent_crossover = float(os.getenv("PERCENT_CROSSOVER", "${PERCENT_CROSSOVER}"))

wandb.config.model_name = se_gym.config.MODEL_CONFIG["model_name"]
wandb.config.population_size = len(INITIAL_θ)
wandb.config.percent_elite = percent_elite
wandb.config.percent_mutation = percent_mutation
wandb.config.percent_crossover = percent_crossover

parquet_path = f"data.{int(time.time())}.parquet"
print(f"Data will be stored in {parquet_path}")

# Define the sampler
π = se_gym.Sampler(
    store=se_gym.observe.Store(
        converter="py",
        retriever="codemap",
    )
)

# Initialize the population
population = se_gym.genetic.Population(
    initial_individuals=INITIAL_θ,
    percent_elite=percent_elite,
    percent_mutation=percent_mutation,
    percent_crossover=percent_crossover,
    sampler=π,
)

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
            s_t = env.reset(issue)
            individual_log = {
                "epoch": epoch,
                "issue": issue,
                "individual": population.individuals.index(individual),
                "steps": []
            }
            r_ind = []
            for timestep in range(MAX_TIME_STEPS):
                print(f"\t\t\tTimestep {timestep}")
                starttime = time.time()
                a_t = population.get_action(individual, s_t)
                s_t = env.step(a_t, s_t)
                r_ind_t = R(s_t)
                r_ind.append(r_ind_t)
                wandb.log({
                    "step": timestep,
                    "score": r_ind_t,
                    "patch": a_t,
                    "individual": population.individuals.index(individual)+1,
                }, commit=False)

                se_gym.utils.log_to_parqet(log_filename=parquet_path,model=se_gym.config.MODEL_CONFIG["model_name"],epoch=epoch,individual_i=population.individuals.index(individual),individual=individual,issue=issue,timestep=timestep,patch=a_t,score=r_ind_t,time=time.time()-starttime)
                if r_ind_t == 1:
                    print(f"\t\t\t\tIssue fixed in {timestep} timesteps")
                    wandb.log({"issue_fixed": True})
                    break
            else:
                print(f"\t\t\tIssue not fixed in {timestep} timesteps")
                wandb.log({"issue_fixed": False})
            wandb.log(individual_log, commit=False)
            rewards.append(r_ind)
        epoch_loss.append(rewards)
        wandb.log({"epoch": epoch, "issue": issue}, commit=True)

    epoch_loss = list(map(list, zip(*epoch_loss)))
    population.evolve(epoch_loss)

wandb.finish()
EOF