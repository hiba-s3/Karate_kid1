import time
import numpy as np
import cv2 as cv
import src.moves_manager as moves
import src.screen_capture as screenshot
import src.keys_manager as keys
import src.image_processing as imageP
import src.game_algorithms as game

def initialize_game():
    """Initialize the game setup and return the game window region."""
    region = screenshot.init_screenshot()
    if region is None:
        print("Game window not found")
        return None
    imageP.clean_folder()
    for _ in range(3):
        moves.saveMove("right")
    keys.trigger_event(moves.getMove())
    time.sleep(0.1)
    return region

def process_iteration(region, count, process_glass, interval, iteration, temporary_numb, remember):
    """Process a single iteration of the game loop."""
    start_time = time.time()

    # Only capture the screen on every second count
    if count % 1 == 0:
        game_window = screenshot.capture_game_window(region, count)
        lantern = imageP.lantern_detection(game_window,count)
        if imageP.detect_game_over(game_window):
            return False, count, process_glass, temporary_numb, remember

        if count > 1:
            trunk_image = imageP.isolate_trunk(game_window, count)
            temp_glass = game.detect_glass(trunk_image, count)
        else:
            trunk_image = imageP.isolate_first_trunk(game_window, count)
            temp_glass = False

        rate = game.calculate_white_pixel_rate(trunk_image)
        print(f"Rate of white pixels: {rate:.2f}%")
        if rate < 70:
            numb = game.detectIFisnum(trunk_image)
        else:
            numb = 0 
        trunk_image = game.prepro(trunk_image, numb)
        if numb != 0:
            print(f"number {numb} in step {count}")
        if temp_glass:
            process_glass += 1
        glass = False
        if process_glass == 2:
            print(f"detected glass at step {count}")
            process_glass = 0
            glass = temp_glass

        position = game.process(trunk_image, count, temp_glass, glass)
        if position == "none":
            print(f"Error in position in iteration {iteration}")
            return True, count, process_glass, temporary_numb, remember
        if lantern:
            position = lantern
        # Save two of the same moves for the position
        # moves.saveMove(position)
        moves.saveMove(position)
        remember[count] = 1

        if numb < 50 and numb > 0:
            temporary_numb += 1
            if temporary_numb == 2:
                if numb > 5:
                    numb = numb / 10
                for _ in range(int(numb-1)):
                    moves.saveMove(position)
                    remember[count] += 1
                temporary_numb = 0
        else:
            temporary_numb = 0
        if glass:
            moves.saveMove(position)
            remember[count] += 1

    # Trigger two moves for each screen capture
    if count > 1:
        if count > 3:
            print(f"remember in count {count} is {remember[count - 3]}")
            for _ in range(remember[count - 3]):  # Use remembered iterations
                move = moves.getMove()
                if move:
                    print(f"The move number {count} is {move}")
                    keys.trigger_event(move)
        else:
            for _ in range(1):
                move = moves.getMove()
                if move:
                    print(f"The move number {count} is {move}")
                    keys.trigger_event(move)

    elapsed_time = time.time() - start_time
    time_to_sleep = max(0, interval - elapsed_time)
    if time_to_sleep == 0:
        print("Processing time exceeded the interval limit")
    time.sleep(time_to_sleep)
    return True, count + 1, process_glass, temporary_numb, remember

def main():
    """Main function to execute the Karate Kiddo Auto Player."""
    print("Welcome to Karate Kiddo Auto Player!!")
    keys.waitKey()
    region = initialize_game()
    if region is None:
        return
    count = 1
    process_glass = 0
    iteration = 0
    interval = 0.4
    temporary_numb = 0
    remember = [0] * 1000 
    while True:
        iteration += 1
        success, count, process_glass, temporary_numb, remember = process_iteration(region, count, process_glass, interval, iteration, temporary_numb, remember)
        if not success:
            break

if __name__ == "__main__":
    main()
