import sqlite3
import os

def check_users():
    if not os.path.exists('hydroalert.db'):
        print("No database file found")
        return
    
    try:
        conn = sqlite3.connect('hydroalert.db')
        cursor = conn.cursor()
        
        # Check what tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print("Tables in database:", tables)
        
        # Check if users table exists and has data
        if ('users',) in tables:
            cursor.execute("SELECT * FROM users")
            users = cursor.fetchall()
            print(f"\nFound {len(users)} users:")
            for user in users:
                print(f"  User ID: {user[0]}")
                if len(user) > 1:
                    print(f"  Username: {user[1]}")
                if len(user) > 2:
                    print(f"  Email: {user[2]}")
                print()
        else:
            print("No users table found")
            
        conn.close()
        
    except Exception as e:
        print(f"Error reading database: {e}")

if __name__ == "__main__":
    check_users()
