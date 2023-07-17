import pygame
import time
import random

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
    num1 = random.randint(1, 10)
    num2 = random.randint(1, 10)
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
        # Generate a new question
        question, answer = generate_question()
        total_questions += 1

        # Display the question
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
            
            if not in_calculation_game and not in_meditation:
                start_time = time.time()
                in_calculation_game = True

    
    elapsed_time = time.time() - start_time
    if in_calculation_game:
        if elapsed_time >= game_duration:
            in_calculation_game = False
            in_meditation = True
            start_time = time.time()
            run_calculation_game()  
        else:
            
            window.fill((255, 255, 255))
            time_remaining = game_duration - elapsed_time
            time_text = font.render(f"Time Remaining: {int(time_remaining)} seconds", True, (0, 0, 0))
            window.blit(time_text, (width // 2 - time_text.get_width() // 2, height // 2 - time_text.get_height() // 2))
            pygame.display.flip()

    elif in_meditation:
        elapsed_time = time.time() - start_time
        if elapsed_time >= game_duration:
            window.fill((0, 0, 0))
            message_text = font.render("Please Close Your Eyes and Take Deep Breaths", True, (255, 255, 255))
            window.blit(message_text, (width // 2 - message_text.get_width() // 2, height // 2 - message_text.get_height() // 2))
            pygame.display.flip()
            time.sleep(60)  
            running = False
            
pygame.quit()
