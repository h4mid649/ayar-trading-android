[app]

# نام اپلیکیشن
title = AYAR Trading

# نام پکیج (باید یکتا باشد)
package.name = ayartrading

# دامنه پکیج
package.domain = com.ayar

# دایرکتوری سورس
source.dir = .

# فایل اصلی
source.include_exts = py,png,jpg,kv,atlas,json

# نسخه
version = 1.0.0

# نیازمندی‌ها
requirements = python3,kivy,requests

# ماژول‌های اضافی
# android.permissions = INTERNET,ACCESS_NETWORK_STATE

# آیکون (اختیاری)
# icon.filename = %(source.dir)s/assets/icon.png

# Orientation
orientation = portrait

# Fullscreen
fullscreen = 0

# اندروید
android.permissions = INTERNET,ACCESS_NETWORK_STATE,WRITE_EXTERNAL_STORAGE
android.api = 31
android.minapi = 21
android.ndk = 25b
android.accept_sdk_license = True

# معماری
android.archs = arm64-v8a,armeabi-v7a

[buildozer]

# مسیر log
log_level = 2

# هشدارها
warn_on_root = 1
