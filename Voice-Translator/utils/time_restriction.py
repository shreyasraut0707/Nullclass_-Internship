# -*- coding: utf-8 -*-
"""
Time Restriction Utility for Voice Translator
Controls when the translation service is available
Operating hours: 9:30 PM to 10:00 PM (21:30 to 22:00)
"""

from datetime import datetime, time


class TimeRestriction:
    """
    Manages time-based access control for the voice translator
    The service is only active during specified hours
    """
    
    def __init__(self, start_hour=21, start_minute=0, end_hour=22, end_minute=0):
        """
        Initialize time restriction with operating hours
        
        Args:
            start_hour: Start hour in 24-hour format (default: 21 for 9:00 PM)
            start_minute: Start minute (default: 0)
            end_hour: End hour in 24-hour format (default: 22 for 10:00 PM)
            end_minute: End minute (default: 0)
        """
        self.start_time = time(start_hour, start_minute, 0)
        self.end_time = time(end_hour, end_minute, 0)
        
        # Messages
        self.REST_MESSAGE = "Taking rest, see you tomorrow!"
        self.ACTIVE_MESSAGE = "Translator is active and ready!"
        
    def is_service_active(self):
        """
        Check if current time is within operating hours
        
        Returns:
            bool: True if service is active, False otherwise
        """
        current_time = datetime.now().time()
        
        # Check if current time is within the allowed window
        if self.start_time <= current_time <= self.end_time:
            return True
        return False
    
    def get_current_status(self):
        """
        Get detailed status of the service
        
        Returns:
            dict: Status information including active state and message
        """
        is_active = self.is_service_active()
        current_time = datetime.now()
        
        status = {
            'is_active': is_active,
            'current_time': current_time.strftime("%I:%M %p"),
            'operating_hours': f"{self.start_time.strftime('%I:%M %p')} - {self.end_time.strftime('%I:%M %p')}",
            'message': self.ACTIVE_MESSAGE if is_active else self.REST_MESSAGE
        }
        
        # Calculate time until next active period
        if not is_active:
            status['time_until_active'] = self._calculate_time_until_active()
        else:
            status['time_remaining'] = self._calculate_time_remaining()
        
        return status
    
    def _calculate_time_until_active(self):
        """Calculate time until service becomes active"""
        now = datetime.now()
        current_time = now.time()
        
        # If before start time today
        if current_time < self.start_time:
            start_datetime = datetime.combine(now.date(), self.start_time)
            delta = start_datetime - now
        else:
            # After end time, calculate for tomorrow
            from datetime import timedelta
            tomorrow = now.date() + timedelta(days=1)
            start_datetime = datetime.combine(tomorrow, self.start_time)
            delta = start_datetime - now
        
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, _ = divmod(remainder, 60)
        
        return f"{hours} hours, {minutes} minutes"
    
    def _calculate_time_remaining(self):
        """Calculate time remaining in active period"""
        now = datetime.now()
        end_datetime = datetime.combine(now.date(), self.end_time)
        
        if end_datetime < now:
            return "0 minutes"
        
        delta = end_datetime - now
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, _ = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours} hours, {minutes} minutes"
        return f"{minutes} minutes"
    
    def get_status_message(self):
        """
        Get appropriate message based on current time
        
        Returns:
            str: Status message for display
        """
        if self.is_service_active():
            return self.ACTIVE_MESSAGE
        return self.REST_MESSAGE
    
    def format_operating_hours(self):
        """Get formatted operating hours string"""
        start_str = self.start_time.strftime("%I:%M %p")
        end_str = self.end_time.strftime("%I:%M %p")
        return f"{start_str} to {end_str}"
    
    def set_operating_hours(self, start_hour, start_minute, end_hour, end_minute):
        """
        Update operating hours
        
        Args:
            start_hour: New start hour (24-hour format)
            start_minute: New start minute
            end_hour: New end hour (24-hour format)
            end_minute: New end minute
        """
        self.start_time = time(start_hour, start_minute, 0)
        self.end_time = time(end_hour, end_minute, 0)


def test_time_restriction():
    """Test the time restriction module"""
    print("="*50)
    print("TIME RESTRICTION TEST")
    print("="*50)
    
    restriction = TimeRestriction()
    status = restriction.get_current_status()
    
    print(f"\nCurrent Time: {status['current_time']}")
    print(f"Operating Hours: {status['operating_hours']}")
    print(f"Service Active: {'Yes' if status['is_active'] else 'No'}")
    print(f"Message: {status['message']}")
    
    if 'time_until_active' in status:
        print(f"Time until active: {status['time_until_active']}")
    elif 'time_remaining' in status:
        print(f"Time remaining: {status['time_remaining']}")
    
    print("\n" + "="*50)


if __name__ == "__main__":
    test_time_restriction()
