from pynput import keyboard

def on_press(key):
    try:
        # Check if the Page Down key is pressed
        if key == keyboard.Key.page_down:
            # Simulate an Enter key press
            with keyboard.Controller() as controller:
                controller.press(keyboard.Key.enter)
                controller.release(keyboard.Key.enter)
    except Exception as e:
        print(f"Error: {e}")

# Listener
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()
