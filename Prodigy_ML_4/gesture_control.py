import pyautogui
import win32gui


def control(prev, result):
    if result == 'palm' and result != prev and get_active_window_title() == "":  # screen up
        prev = result
        # print("palm")
        desk()

    elif result == 'switch' and result != prev:  # switch app
        if get_active_window_title() == "":
            desk()
            prev = 'palm'
        else:
            switch()
            prev = 'switch'
        # print("switch", prev)

    elif result == 'down' and result != prev and get_active_window_title() != "":  # desktop
        prev = result
        # print("down")
        desk()

    elif result == 'mouse' and prev != 'mouse':
        prev = result

    elif result == 'fist':
        prev = result

    else:
        pass
        # print(prev, "else")
    return prev


def get_active_window_title():
    window = win32gui.GetForegroundWindow()
    return win32gui.GetWindowText(window)


def desk():
    pyautogui.hotkey("win", "d")


def switch():
    pyautogui.hotkey("alt", "esc")
