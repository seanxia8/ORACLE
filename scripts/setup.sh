#!/bin/bash

# Check if current machine is an etp portal machine.
PORTAL_LIST=("bms1" "bms2" "bms3" "bms1-centos7" "bms2-centos7" "bms3-centos7" "portal1")
CURRENT_HOST=$(hostname --short)
if [[ ! " ${PORTAL_LIST[*]} " =~ " ${CURRENT_HOST} " ]]; then  
    echo "Current host (${CURRENT_HOST}) not in list of portal machines with a shedduler:"
    printf '%s\n' "${PORTAL_LIST[@]}"
    echo "You will not be able to test the batch jobs from here."
fi
echo "Running on host: ${CURRENT_HOST}."

# Check if os is supported.
OS_NAME=$( lsb_release -d | sed 's/Description:\s\+\(.*\)/\1/')
OS_ID=$(lsb_release -i | sed 's/.*:\s\+\(.*\)/\1/')
OS_RELEASE=$(lsb_release -r | sed 's/.*:\s\+\(.*\)/\1/')
OS_MAJOR=$(echo ${OS_RELEASE} | sed 's/\([0-9]\+\)\..*/\1/')
ID_LIST=("CentOS" "RedHatEnterprise")
if [[ ! " ${ID_LIST[*]} " =~ " ${OS_ID} " ]]; then
    echo "OS of Current host (${OS_NAME}) not in list of tested OS types:"
    printf '%s\n' "${ID_LIST[@]}"
    echo "There might be problems."
fi
case ${OS_MAJOR} in
    7|8)
	SOURCE_PATH="/cvmfs/etp.kit.edu/GPU_examples/miniconda/bin/activate ML_GPU"
	;;
    *)
        echo "Version ${OS_RELEASE} of ${OS_ID} was not tested before. there might be problems."
        SOURCE_PATH="/cvmfs/etp.kit.edu/GPU_examples/miniconda/bin/activate ML_GPU"
	;;
esac
echo "Running on OS: ${OS_NAME}."

# Get dir of setup script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Save MNIST dataset to current directory
if [[ ! -d "${SCRIPT_DIR}/data_mnist" ]]; then
    (
        source ${SOURCE_PATH}
        python3 ${SCRIPT_DIR}/save_mnist_to.py --location "${SCRIPT_DIR}/data_mnist"
    )
fi

FALLBACK_USER="tvoigtlaender"
USER_PATH="/ceph/srv/${USER}"
if [[ ! -d "${USER_PATH}" ]]; then
    echo "/ceph/srv/ directory of ${USER} does not exist. Using fallback to ${FALLBACK_USER}."
    echo "Please ask an admin to get you own directory."
    USE_USER="${FALLBACK_USER}"
else
    USE_USER="${USER}"
    TARGET_PATH="/ceph/srv/${USE_USER}/for_condor/data_mnist"
    # Save MNIST dataset to /ceph/srv directory
    if [[ ! -d "${TARGET_PATH}" ]]; then
        (
            source ${SOURCE_PATH}
            python3 ${SCRIPT_DIR}/save_mnist_to.py --location ${TARGET_PATH}
        )
    fi
fi
export USE_USER
echo "Setup successful. You can now try any of the local or batch variants running on CPU or GPU."
