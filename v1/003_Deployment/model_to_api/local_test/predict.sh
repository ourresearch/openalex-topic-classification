#!/bin/bash

payload=$1
content=${2:-application/json}

curl -d @${payload} -H "Content-Type: ${content}" -vX POST http://localhost:8080/invocations
