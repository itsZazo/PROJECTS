# scedule_tool.py
import win32com.client
from mcp.server.fastmcp import FastMCP
from datetime import datetime
import pythoncom

mcp = FastMCP("AI Scedule Tool")

@mcp.tool()
def schedule_in_windows_calendar(
    name: str,
    email: str, 
    date: str,
    start_time: str,
    end_time: str
) -> dict:
    """
    Schedule a meeting in Windows Outlook Calendar
    
    Args:
        name: Client name
        email: Client email address
        date: Meeting date in YYYY-MM-DD format
        start_time: Start time in HH:MM format
        end_time: End time in HH:MM format
    
    Returns:
        dict: Success status and message
    """
    
    try:
        # Initialize COM
        pythoncom.CoInitialize()
        
        # Connect to Outlook
        outlook = win32com.client.Dispatch("Outlook.Application")
        
        # Create appointment
        appointment = outlook.CreateItem(1)  # 1 = Appointment
        
        # Set basic properties
        appointment.Subject = f"Meeting with {name}"
        
        # Parse and set datetime
        start_datetime = f"{date} {start_time}"
        end_datetime = f"{date} {end_time}"
        
        # Convert to datetime objects for better handling
        start_dt = datetime.strptime(start_datetime, "%Y-%m-%d %H:%M")
        end_dt = datetime.strptime(end_datetime, "%Y-%m-%d %H:%M")
        
        appointment.Start = start_dt
        appointment.End = end_dt
        appointment.RequiredAttendees = email
        
        # Additional properties
        appointment.Body = f"Meeting scheduled with {name} ({email})"
        appointment.ReminderMinutesBeforeStart = 15
        appointment.ReminderSet = True
        appointment.BusyStatus = 2  # Busy
        
        # Save the appointment
        appointment.Save()
        
        print(f"Meeting scheduled: {name} on {date} from {start_time} to {end_time}")
        
        return {
            "success": True,
            "message": f"Meeting with {name} scheduled successfully in Outlook Calendar",
            "details": {
                "client": name,
                "email": email,
                "date": date,
                "time": f"{start_time} - {end_time}"
            }
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error scheduling meeting: {error_msg}")
        
        return {
            "success": False,
            "error": error_msg,
            "message": "Failed to schedule meeting in Outlook Calendar"
        }
    
    finally:
        # Clean up COM
        try:
            pythoncom.CoUninitialize()
        except:
            pass

if __name__ == "__main__":
    mcp.run(transport="stdio")