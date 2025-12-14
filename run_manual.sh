#!/bin/bash

# set cwd to current file path
cd "$(dirname "$0")"

# get run configuration from the first argument
profile=$1

# shutdown all running containers
docker compose -f docker_compose.yaml --profile $profile down

# apply the rest of the arguments to the python script
shift 1
docker compose -f docker_compose.yaml --profile $profile run -d --build --service-ports "$profile-service" python3 $@

sleep 5