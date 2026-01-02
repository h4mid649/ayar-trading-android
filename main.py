# -*- coding: utf-8 -*-
"""
AYAR Trading - Android UI
رابط کاربری اندروید برای سیستم معاملاتی
"""

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.uix.switch import Switch
from kivy.clock import Clock
from kivy.core.window import Window
import threading
import json
from pathlib import Path

# تنظیم رنگ پس‌زمینه
Window.clearcolor = (0.95, 0.95, 0.95, 1)


class TradingUI(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = 10
        self.spacing = 10
        
        self.is_running = False
        self.current_inscode = "34144395039913458"
        
        # ساخت UI
        self.build_ui()
        
    def build_ui(self):
        # ========== هدر ==========
        header = BoxLayout(size_hint=(1, 0.08), padding=5)
        header_bg = BoxLayout()
        
        title = Label(
            text='[b]AYAR Trading System[/b]',
            markup=True,
            font_size='22sp',
            color=(0.1, 0.3, 0.6, 1)
        )
        header_bg.add_widget(title)
        header.add_widget(header_bg)
        self.add_widget(header)
        
        # ========== تنظیمات ورودی ==========
        settings_box = GridLayout(
            cols=2,
            size_hint=(1, 0.25),
            padding=5,
            spacing=5
        )
        
        # InsCode
        settings_box.add_widget(Label(
            text='InsCode:',
            size_hint_x=0.3,
            color=(0, 0, 0, 1)
        ))
        self.inscode_input = TextInput(
            text=self.current_inscode,
            multiline=False,
            size_hint_x=0.7,
            font_size='16sp',
            background_color=(1, 1, 1, 1),
            foreground_color=(0, 0, 0, 1)
        )
        settings_box.add_widget(self.inscode_input)
        
        # Capital
        settings_box.add_widget(Label(
            text='سرمایه (میلیون):',
            size_hint_x=0.3,
            color=(0, 0, 0, 1)
        ))
        self.capital_input = TextInput(
            text='500',
            multiline=False,
            size_hint_x=0.7,
            input_filter='int',
            font_size='16sp',
            background_color=(1, 1, 1, 1),
            foreground_color=(0, 0, 0, 1)
        )
        settings_box.add_widget(self.capital_input)
        
        # Risk %
        settings_box.add_widget(Label(
            text='ریسک (%):',
            size_hint_x=0.3,
            color=(0, 0, 0, 1)
        ))
        self.risk_input = TextInput(
            text='1.0',
            multiline=False,
            size_hint_x=0.7,
            input_filter='float',
            font_size='16sp',
            background_color=(1, 1, 1, 1),
            foreground_color=(0, 0, 0, 1)
        )
        settings_box.add_widget(self.risk_input)
        
        # Watch Mode
        settings_box.add_widget(Label(
            text='حالت نظارت:',
            size_hint_x=0.3,
            color=(0, 0, 0, 1)
        ))
        self.watch_switch = Switch(
            active=True,
            size_hint_x=0.7
        )
        settings_box.add_widget(self.watch_switch)
        
        self.add_widget(settings_box)
        
        # ========== دکمه‌های کنترل ==========
        controls = BoxLayout(size_hint=(1, 0.1), spacing=10, padding=5)
        
        self.start_btn = Button(
            text='شروع معامله',
            font_size='18sp',
            background_color=(0.2, 0.7, 0.3, 1),
            color=(1, 1, 1, 1),
            bold=True
        )
        self.start_btn.bind(on_press=self.toggle_trading)
        controls.add_widget(self.start_btn)
        
        self.clear_btn = Button(
            text='پاک کردن لاگ',
            font_size='18sp',
            background_color=(0.9, 0.5, 0.1, 1),
            color=(1, 1, 1, 1),
            bold=True
        )
        self.clear_btn.bind(on_press=self.clear_log)
        controls.add_widget(self.clear_btn)
        
        self.add_widget(controls)
        
        # ========== وضعیت ==========
        status_box = BoxLayout(size_hint=(1, 0.08), padding=5)
        self.status_label = Label(
            text='[color=808080]آماده برای شروع...[/color]',
            markup=True,
            font_size='16sp',
            color=(0, 0, 0, 1)
        )
        status_box.add_widget(self.status_label)
        self.add_widget(status_box)
        
        # ========== لاگ نمایش ==========
        scroll = ScrollView(size_hint=(1, 0.49))
        self.log_label = Label(
            text='',
            markup=True,
            font_size='13sp',
            size_hint_y=None,
            color=(0.1, 0.1, 0.1, 1),
            halign='left',
            valign='top',
            padding=(10, 10)
        )
        self.log_label.bind(
            texture_size=lambda *x: setattr(self.log_label, 'height', self.log_label.texture_size[1])
        )
        scroll.add_widget(self.log_label)
        self.add_widget(scroll)
        
    def toggle_trading(self, instance):
        if not self.is_running:
            self.start_trading()
        else:
            self.stop_trading()
    
    def start_trading(self):
        self.is_running = True
        self.start_btn.text = 'توقف معامله'
        self.start_btn.background_color = (0.9, 0.2, 0.2, 1)
        self.update_status('[color=00aa00]● در حال اجرا...[/color]')
        
        # شروع thread معامله
        self.trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
        self.trading_thread.start()
    
    def stop_trading(self):
        self.is_running = False
        self.start_btn.text = 'شروع معامله'
        self.start_btn.background_color = (0.2, 0.7, 0.3, 1)
        self.update_status('[color=ff0000]● متوقف شد[/color]')
    
    def trading_loop(self):
        """حلقه اصلی معامله"""
        try:
            # ایمپورت کد اصلی
            from ayar_trading import run_once, Config, PerformanceTracker
            
            # تنظیمات
            config = Config(
                capital_rial=int(self.capital_input.text) * 1_000_000,
                risk_pct=float(self.risk_input.text)
            )
            
            tracker = PerformanceTracker()
            inscode = self.inscode_input.text.strip()
            
            self.add_log(f"[b]شروع معامله برای InsCode: {inscode}[/b]")
            
            while self.is_running:
                try:
                    # اجرای یک چرخه
                    self.add_log("[color=0066cc]در حال بررسی بازار...[/color]")
                    
                    # فراخوانی تابع اصلی
                    run_once(inscode, config, tracker)
                    
                    self.add_log("[color=00aa00]✓ چرخه کامل شد[/color]")
                    
                    # اگر watch mode فعال نیست، فقط یکبار اجرا شود
                    if not self.watch_switch.active:
                        break
                    
                    # تاخیر 30 ثانیه
                    import time
                    for i in range(30):
                        if not self.is_running:
                            break
                        time.sleep(1)
                    
                except Exception as e:
                    self.add_log(f"[color=ff0000]خطا: {str(e)}[/color]")
                    import time
                    time.sleep(10)
            
            self.add_log("[b]معامله متوقف شد.[/b]")
            
        except Exception as e:
            self.add_log(f"[color=ff0000]خطای اصلی: {str(e)}[/color]")
        finally:
            Clock.schedule_once(lambda dt: self.stop_trading(), 0)
    
    def add_log(self, message):
        """افزودن پیام به لاگ"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        def update_ui(dt):
            current = self.log_label.text
            new_line = f"[{timestamp}] {message}"
            
            # محدود کردن تعداد خطوط
            lines = current.split('\n')
            if len(lines) > 100:
                lines = lines[-100:]
            
            lines.append(new_line)
            self.log_label.text = '\n'.join(lines)
        
        Clock.schedule_once(update_ui, 0)
    
    def update_status(self, message):
        """بروزرسانی وضعیت"""
        def update_ui(dt):
            self.status_label.text = message
        Clock.schedule_once(update_ui, 0)
    
    def clear_log(self, instance):
        """پاک کردن لاگ"""
        self.log_label.text = ''
        self.add_log('[color=808080]لاگ پاک شد[/color]')


class AyarTradingApp(App):
    def build(self):
        self.title = 'AYAR Trading'
        return TradingUI()


if __name__ == '__main__':
    AyarTradingApp().run()
