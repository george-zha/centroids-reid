# Configure Linux for Neuron repository updates
. /etc/os-release
sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
EOF
wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -

# Update OS packages
sudo apt-get update -y

################################################################################################################
# To install or update to Neuron versions 1.19.1 and newer from previous releases:
# - DO NOT skip 'aws-neuron-dkms' install or upgrade step, you MUST install or upgrade to latest Neuron driver
################################################################################################################

# Install OS headers
sudo apt-get install linux-headers-$(uname -r) -y

# Install Neuron Driver
sudo apt-get install aws-neuronx-dkms -y

####################################################################################
# Warning: If Linux kernel is updated as a result of OS package update
#          Neuron driver (aws-neuron-dkms) should be re-installed after reboot
####################################################################################

# Install Neuron Tools
sudo apt-get install aws-neuronx-tools -y

export PATH=/opt/aws/neuron/bin:$PATH

######################################################
#   Only for Ubuntu 20 - Install Python3.7
#
# sudo add-apt-repository ppa:deadsnakes/ppa
# sudo apt-get install python3.7
#
######################################################
# Install Python venv and activate Python virtual environment to install    
# Neuron pip packages.
sudo apt-get install -y python3.7-venv g++
python3.7 -m venv pytorch_venv
source pytorch_venv/bin/activate
pip install -U pip


# Instal Jupyter notebook kernel 
pip install ipykernel 
python3.7 -m ipykernel install --user --name pytorch_venv --display-name "Python (Neuron PyTorch)"
pip install jupyter notebook
pip install environment_kernels


# Set Pip repository  to point to the Neuron repository
pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

#Install Neuron PyTorch
pip install torch-neuron neuron-cc[tensorflow] "protobuf==3.20.1" torchvision