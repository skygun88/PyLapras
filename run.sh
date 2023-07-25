# Format : bash run.sh LaprasAgent $AgentName SpaceName/configFile
# Format : bash run.sh $LaprasAgent $AgentName SpaceName/configFile

AGENT_SPACES=("LaprasAgent" "Hejhome" "Simulation")
AGENT_SPACE_PATHS=("agent" "agent/Hejhome" "agent/Simulator")
SPACE_LEN="${#AGENT_SPACE_PATHS[@]}" 

# Check the number of Arguments 
if [ "$#" -lt 2 ]
then
    echo "Need more arguments"
    exit 0
else
    space_name="$1"
    agent_name="$2"
    # Check space name
    for ((i=0; i<SPACE_LEN; i++))
    do
        if [ "${space_name}" == "${AGENT_SPACES[i]}" ]
        then 
            space_path="${AGENT_SPACE_PATHS[i]}"
        fi
    done
    if [ -z "${space_path}" ]
    then
        echo "Wrong first augment - Need one of [${AGENT_SPACES[*]}]" 
        exit 0
    fi
fi

echo "Space - ${space_name} | Agent - ${agent_name}"

BASEDIR="$( cd "$( dirname "$0" )" && pwd )"
AGENT_PATH="${BASEDIR}/${space_path}/${agent_name}.py"

if [ ! -f "${AGENT_PATH}" ]
then
    echo "There is no target agent in the given space - ${AGENT_PATH}"
    exit 0
fi
echo "File Detected - ${AGENT_PATH}"


# while getopts "c:h" opt
# do
#     echo "${opt}"
#     case "${opt}" in
#         "c") CUDA_OPTION="CUDA_VISIBLE_DEVICES=${OPTARG}"; echo "${CUDA_OPTION}";;
#         "h") echo "bash run.sh \{space_name\} \{agent_name\} | \{config_path\} | -c \{CUDA_VISIBLE_DEVICES\}"; exit 0;;
#     esac 
# done



if [ "$#" -eq 3 ]
then
    CONFIG_PATH=="${BASEDIR}/resources/$3"
    if [ ! -f "${CONFIG_PATH}" ]
    then
        echo "There is no target configuration - ${CONFIG_PATH}"
        exit 0
    fi
    ${CUDA_OPTION} python ${AGENT_PATH} -c ${CONFIG_PATH}
else
    ${CUDA_OPTION} python ${AGENT_PATH}
fi