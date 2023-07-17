import pygame
import time
import random
import subprocess


pygame.init()


width, height = 800, 600
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Calculation Game and Meditation")


start_time = 0
game_duration = 60  


in_calculation_game = True
in_meditation = False


font = pygame.font.Font(None, 36)


def generate_question():
    num1 = random.randint(100, 10060)
    num2 = random.randint(100, 1000)
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
    print("Running calculation game...")
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

            # Update the display with the user's answer
            window.fill((255, 255, 255))
            question_text = font.render(question, True, (0, 0, 0))
            window.blit(question_text, (width // 2 - question_text.get_width() // 2, height // 2 - question_text.get_height() // 2))
            answer_text = font.render(user_answer, True, (0, 0, 0))
            window.blit(answer_text, (width // 2 - answer_text.get_width() // 2, height // 2 + question_text.get_height()))
            pygame.display.flip()

        # Check if the answer is correct
        if int(user_answer) == answer:
            correct_answers += 1

        # Generate a new question after a short delay
        time.sleep(1)

        # Calculate the elapsed time
        elapsed_time = time.time() - start_time

    # Game over
    print(f"Game Over!\nTotal Questions: {total_questions}\nCorrect Answers: {correct_answers}")



running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
   
            if not in_calculation_game and not in_meditation:
                start_time = time.time()
                in_calculation_game = True


    if in_calculation_game:
        elapsed_time = time.time() - start_time
        if elapsed_time >= game_duration:
            in_calculation_game = False
            in_meditation = True
            start_time = time.time()
            run_calculation_game() 
    elif in_meditation:
        elapsed_time = time.time() - start_time
        if elapsed_time >= game_duration:
            in_meditation = False
            play_meditation_video()  


    window.fill((255, 255, 255))
    pygame.display.flip()


pygame.quit()
