def locate_from_health(score):
    if score < 8.4:
        return "Normal"

    elif score < 8.75:
        return "Ball"

    elif score < 9:
        return "Inner Race"
    
    else:
        return "Outer Race"