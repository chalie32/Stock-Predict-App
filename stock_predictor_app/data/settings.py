import json
import os
from pathlib import Path

class Settings:
    def __init__(self):
        self._default_settings = {
            "Chart Settings": {
                "chart_type": "Candlestick",
                "default_ma": {
                    "MA20": True,
                    "MA50": True,
                    "MA200": False
                },
                "show_volume": True
            },
            "Analysis Settings": {
                "default_timeframe": "1 Year",
                "default_model": "LSTM",
                "prediction_days": 30
            },
            "Alert Settings": {
                "price_alerts": True,
                "price_change_threshold": 5
            }
        }
        
        self.settings_file = Path("settings.json")
        self.settings = self.load_settings()
    
    def load_settings(self):
        """Load settings from file or create with defaults if file doesn't exist"""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                # Merge with defaults to ensure all settings exist
                return self._merge_with_defaults(settings)
            return self._default_settings.copy()
        except Exception as e:
            print(f"Error loading settings: {e}")
            return self._default_settings.copy()
    
    def _merge_with_defaults(self, settings):
        """Merge loaded settings with defaults to ensure all settings exist"""
        merged = self._default_settings.copy()
        for category in merged:
            if category in settings:
                for key in merged[category]:
                    if key in settings[category]:
                        if isinstance(merged[category][key], dict):
                            # For nested settings like default_ma
                            merged[category][key].update(settings[category][key])
                        else:
                            merged[category][key] = settings[category][key]
        return merged
    
    def save_settings(self):
        """Save current settings to file"""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def get_setting(self, category, key):
        """Get a specific setting value"""
        try:
            return self.settings[category][key]
        except KeyError:
            # If setting doesn't exist, return default
            return self._default_settings.get(category, {}).get(key)
    
    def update_setting(self, category, key, value):
        """Update a specific setting value"""
        if category not in self.settings:
            self.settings[category] = {}
        self.settings[category][key] = value
        self.save_settings()
    
    def reset_to_defaults(self):
        """Reset all settings to default values"""
        self.settings = self._default_settings.copy()
        self.save_settings()
    
    def get_all_settings(self):
        """Get all settings"""
        return self.settings 