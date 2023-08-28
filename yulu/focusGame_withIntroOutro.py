import pygame
import time
import pygame.mixer
import random

pygame.init()
pygame.mixer.init()

width, height = 800, 600
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Focus Experiment")

font = pygame.font.Font(None, 36)

game_duration = 60
intro_outro_duration = 20
start_time = time.time()

state = "intro"
pygame.mixer.music.load("calming_audio.mp3")

def generate_question():
    num1 = random.randint(10, 100)
    num2 = random.randint(10, 100)
    operator = random.choice(["+", "-", "*", "/"])
    question = f"{num1} {operator} {num2}"
    if operator == "+":
        answer = num1 + num2
    elif operator == "-":
        answer = num1 - num2
    elif operator == "*":
        answer = num1 * num2
    else:
        answer = num1 // num2
    return question, answer

def run_calculation_game():
    correct_answers = 0
    total_questions = 0
    elapsed_time = 0
    while elapsed_time < game_duration:
        question, answer = generate_question()
        total_questions += 1

        window.fill((255, 255, 255))
        question_text = font.render(question, True, (0, 0, 0))
        window.blit(question_text, (width // 2 - question_text.get_width() // 2, height // 2 - question_text.get_height() // 2))
        pygame.display.flip()

        answer_entered = False
        user_answer = ""
        while not answer_entered:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        answer_entered = True
                    elif event.key == pygame.K_BACKSPACE:
                        user_answer = user_answer[:-1]
                    else:
                        user_answer += event.unicode

            window.fill((255, 255, 255))
            question_text = font.render(question, True, (0, 0, 0))
            window.blit(question_text, (width // 2 - question_text.get_width() // 2, height // 2 - question_text.get_height() // 2))
            answer_text = font.render(user_answer, True, (0, 0, 0))
            window.blit(answer_text, (width // 2 - answer_text.get_width() // 2, height // 2 + question_text.get_height()))
            pygame.display.flip()

        if int(user_answer) == answer:
            correct_answers += 1

        time.sleep(1)
        elapsed_time = time.time() - start_time

    print(f"Game Over!\nTotal Questions: {total_questions}\nCorrect Answers: {correct_answers}")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if state == "intro":
                state = "calculation_game"
                start_time = time.time()

    elapsed_time = time.time() - start_time

    window.fill((0, 0, 0))

    if state == "intro":
        intro_text = font.render("Experiment Begins Now", True, (255, 255, 255))
        window.blit(intro_text, (width // 2 - intro_text.get_width() // 2, height // 2 - intro_text.get_height() // 2))
        if elapsed_time >= intro_outro_duration:
            state = "calculation_game"
            start_time = time.time()

    elif state == "calculation_game":
        if elapsed_time < game_duration + intro_outro_duration:
            run_calculation_game()
        else:
            state = "meditation"
            start_time = time.time()

    elif state == "meditation":
        if elapsed_time < game_duration + intro_outro_duration * 2:
            pygame.mixer.music.play()
            meditation_text = font.render("Please Close Your Eyes", True, (255, 255, 255))
            window.blit(meditation_text, (width // 2 - meditation_text.get_width() // 2, height // 2 - meditation_text.get_height() // 2))
        elif elapsed_time < game_duration + intro_outro_duration * 2 + 60:
            pygame.mixer.music.stop()
            state = "outro"
            start_time = time.time()

    elif state == "outro":
        if elapsed_time < game_duration + intro_outro_duration * 3:
            outro_text = font.render("Experiment is Now Over", True, (255, 255, 255))
            window.blit(outro_text, (width // 2 - outro_text.get_width() // 2, height // 2 - outro_text.get_height() // 2))

    pygame.display.flip()

pygame.quit()
