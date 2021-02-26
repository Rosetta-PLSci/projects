import random


def play():
    user = input("(R) for rock, (P) for paper, (S) for scissors: ").upper()
    computer = random.choice(["R", "P", "S"])
    # .choice([]) ile rastgele seçilene seçenekler belirlenir
    if user == computer:
        return "TIE"

    if win(user, computer):
        print(computer)
        return "YOU WON"

    return "YOU LOST!"


def win(player, opponent):
    if (player == "R" and opponent == "S") or (player == "S" and opponent == "P") \
            or (player == "P" and opponent == "R"):
        return True


print(play())
