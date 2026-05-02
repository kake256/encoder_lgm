import requests
import sys

def send_discord_notification(message):
    # Discord Webhook URL
    webhook_url = "https://discord.com/api/webhooks/1471815304530366599/jtnO_oJz3B5T16v-DTzcVG6sqGmIHvyBuhHYmmP6PEWlfa09fJFAPIoltiyxRc4CruYr"
    
    # Discordは "content" キーを使用
    payload = {"content": message}
    
    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        print("Notification sent successfully!")
    except Exception as e:
        print(f"Failed to send notification: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 send_notification.py [status] [dir_path]")
        sys.exit(1)

    status = sys.argv[1]
    dir_path = sys.argv[2]
    
    if status == "success":
        msg = f"✅ **学習完了**\n不足クラスの生成が正常に終了しました。\nディレクトリ: `{dir_path}`"
    else:
        msg = f"❌ **学習エラー発生**\nプログラムが途中で停止しました。ログを確認してください。\nディレクトリ: `{dir_path}`"
    
    send_discord_notification(msg)