# راهنمای ساخت APK

## پیش‌نیازها

نصب Ubuntu/Linux (توصیه می‌شود Ubuntu 20.04 یا بالاتر)

## مرحله 1: نصب ابزارها

```bash
# بروزرسانی سیستم
sudo apt update && sudo apt upgrade -y

# نصب Python
sudo apt install python3 python3-pip -y

# نصب ابزارهای ساخت
sudo apt install -y \
    build-essential \
    git \
    zip \
    unzip \
    openjdk-11-jdk \
    autoconf \
    libtool \
    pkg-config \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libtinfo5 \
    cmake \
    libffi-dev \
    libssl-dev

# نصب Buildozer
pip3 install --upgrade buildozer
pip3 install --upgrade cython
