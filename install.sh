# install tools
sudo apt update
sudo apt install git -y
sudo apt install fuse -y
sudo apt install rclone -y
sudo apt install python3-pip -y
sudo apt install python3-virtualenv -y


# create mount dir
mkdir ~/gdrive


# copy rclone.conf to ~/.config/rclone/rclone.conf


# mount to gdrive
rclone mount gdrive: ~/gdrive --vfs-cache-mode


# create virtualenv
virtualenv .env