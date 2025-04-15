from controller import Robot, DistanceSensor, Motor, GPS, Supervisor
import numpy as np
import random
import math
import os


class KheperaController:
    def __init__(self):
        # Initialize robot as supervisor
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())

        # Get robot node for position reset
        self.robot_node = self.robot.getSelf()
        self.translation_field = self.robot_node.getField('translation')
        self.rotation_field = self.robot_node.getField('rotation')

        # Starting position and rotation
        self.start_position = [1.125, -1.125, 0]
        self.start_rotation = [0, 0, 1, 3.14159]  # Looking left

        # Initialize motors
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

        # Initialize only front sensors
        self.sensors = []
        self.sensors.append(self.robot.getDevice('ds3'))
        self.sensors.append(self.robot.getDevice('ds4'))
        self.sensors.append(self.robot.getDevice('ds2'))
        self.sensors.append(self.robot.getDevice('ds5'))
        for sensor in self.sensors:
            sensor.enable(self.timestep)

        # Initialize GPS
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)

        # Compass
        self.compass = self.robot.getDevice('compass')
        self.compass.enable(self.timestep)

        # Q-learning parameters
        self.learning_rate = 0.2  # Learning speed
        self.discount_factor = 0.95  # future achievments factor
        self.epsilon = 0.3


        # Goal position
        self.goal_position = (-1.0, 1.0)
        self.prev_distance_to_goal = None

        # Training parameters
        self.total_steps = 0
        self.steps_in_episode = 0
        self.max_steps_per_episode = 500
        self.collisions = 0

        # Define states and actions
        self.map_size = 12
        self.n_states = self.map_size * self.map_size # 12 * 12 = 144 states
        self.n_actions = 4  # 4 действия
        self.q_table = np.zeros((self.n_states, self.n_actions))
        self.state_history = []
        self.action_history = []


        # Loading q table
        if not self.load_q_table():
            print("Creating new Q-table")

        # Movement parameters
        self.base_speed = 10
        self.turn_speed = 6
        self.episode_changes = {}

        self.prev_action = None
        self.consecutive_back_moves = 0

    def save_q_table(self, filename='q_table.npy'):
        try:
            self.q_table = np.clip(self.q_table, -10, 10)
            self.q_table = np.round(self.q_table, 2)
            np.save(filename, self.q_table)
            print(f"Q-table saved to {filename}")
        except Exception as e:
            print(f"Error saving Q-table: {e}")

    def load_q_table(self, filename='q_table.npy'):
        try:
            if os.path.exists(filename):
                self.q_table = np.load(filename)
                print(f"Q-table loaded from {filename}")
                return True
            return False
        except Exception as e:
            print(f"Error loading Q-table: {e}")
            return False

    def update_q_table(self, filename='q_table.npy'):
        try:
            if os.path.exists(filename):
                existing_q_table = np.load(filename)
                mask = self.q_table != 0
                existing_q_table[mask] = self.q_table[mask]
                np.save(filename, existing_q_table)
            else:
                self.save_q_table(filename)
        except Exception as e:
            print(f"Error updating Q-table: {e}")

    def reset_position(self):
        self.translation_field.setSFVec3f(self.start_position)
        self.rotation_field.setSFRotation(self.start_rotation)
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        self.robot.step(self.timestep * 10)

    def get_reward(self):
        current_distance = self.get_distance_to_goal()
        reward = 0
        is_terminal = False

        if current_distance < 0.2:  # Goal reached
            reward = 10.0
            is_terminal = True
        elif max([s.getValue() for s in self.sensors]) > 1000:  # Collision
            reward = -10.0
            is_terminal = True
        else:
            reward = -0.01  # Small penalty for each step
            if self.prev_distance_to_goal is not None:
                distance_change = self.prev_distance_to_goal - current_distance
                reward += distance_change * 2
            if np.all(self.q_table[self.get_state()] == 0):
                reward += 0.5

        self.prev_distance_to_goal = current_distance
        return round(np.clip(reward, -10, 10), 2), is_terminal

    def get_state(self):
        pos = self.gps.getValues()
        pos_x = min(int((pos[0] + 1.5) / 0.25), 11)
        pos_y = min(int((pos[1] + 1.5) / 0.25), 11)
        state = pos_x + pos_y * 12
        return state

    def get_current_direction(self):
        north = self.compass.getValues()
        angle = math.degrees(math.atan2(north[1], north[0]))
        if angle < 0:
            angle += 360.0

        # Choosing Angle
        if 45 <= angle < 135:
            return 90  # East
        elif 135 <= angle < 225:
            return 180  # South
        elif 225 <= angle < 315:
            return 270  # West
        else:
            return 0  # North

    def is_centered(self):
        pos = self.gps.getValues()
        x_error = abs((pos[0] + 1.5) % 0.25 - 0.125)
        y_error = abs((pos[1] + 1.5) % 0.25 - 0.125)
        return x_error < 0.05 and y_error < 0.05

    def get_action(self, state):
        if random.random() < self.epsilon:
            action = random.randint(0, self.n_actions - 1)
        else:
            current_action = np.argmax(self.q_table[state])
            if self.prev_action is not None:
                if abs(self.q_table[state][current_action] - self.q_table[state][self.prev_action]) < 0.5:
                    action = self.prev_action
                else:
                    action = current_action
            else:
                action = current_action

        self.prev_action = action
        return action

    def is_in_cell_center(self, tolerance=0.15):
        pos = self.gps.getValues()
        current_x = (pos[0] + 1.5) / 0.25
        current_y = (pos[1] + 1.5) / 0.25

        # Ищем ближайший центр клетки (x.5, y.5)
        target_x = round(current_x - 0.5) + 0.5
        target_y = round(current_y - 0.5) + 0.5

        x_error = abs(current_x - target_x)
        y_error = abs(current_y - target_y)

        # print(f"Current position: ({current_x:.3f}, {current_y:.3f})")
        # print(f"Target center: ({target_x:.3f}, {target_y:.3f})")
        # print(f"Errors: x={x_error:.3f}, y={y_error:.3f}")

        return x_error < tolerance and y_error < tolerance

    def get_current_angle(self):
        north = self.compass.getValues()
        angle = math.degrees(math.atan2(north[1], north[0]))
        if angle < 0:
            angle += 360.0
        return angle

    def take_action(self, action):
        if not self.is_in_cell_center():
            # print("Moving to center")
            self.left_motor.setVelocity(self.base_speed * 0.7)
            self.right_motor.setVelocity(self.base_speed * 0.7)
            self.robot.step(self.timestep * 100)
            return

        # print("In center, performing action")
        current_state = self.get_state()
        self.state_history.append(current_state)
        self.action_history.append(action)

        if len(self.state_history) > 3:
            self.state_history.pop(0)
            self.action_history.pop(0)

        target_directions = {
            0: 0,  # North
            1: 90,  # East
            2: 180,  # South
            3: 270  # West
        }

        current_angle = self.get_current_angle()
        target_angle = target_directions[action]

        # print(f"Current angle: {current_angle:.1f}, Target: {target_angle}")

        angle_diff = (target_angle - current_angle) % 360
        turn_direction = 1 if angle_diff <= 180 else -1

        while True:
            current_angle = self.get_current_angle()
            angle_diff = (target_angle - current_angle) % 360
            if angle_diff > 180:
                angle_diff -= 360

            if abs(angle_diff) < 0.5:
                break

            self.left_motor.setVelocity(self.turn_speed * turn_direction)
            self.right_motor.setVelocity(-self.turn_speed * turn_direction)
            self.robot.step(self.timestep)

        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        self.robot.step(self.timestep * 10)

        self.left_motor.setVelocity(self.base_speed)
        self.right_motor.setVelocity(self.base_speed)
        self.robot.step(self.timestep * 100)

    def get_distance_to_goal(self):
        pos = self.gps.getValues()
        distance = math.sqrt((pos[0] - self.goal_position[0]) ** 2 +
                             (pos[1] - self.goal_position[1]) ** 2)
        if self.steps_in_episode % 100 == 0:
            print(f"Robot at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), Goal at {self.goal_position}")
            print(f"Distance: {distance:.2f}")
        return distance

    def get_direction_to_goal(self):
        pos = self.gps.getValues()
        angle = math.atan2(self.goal_position[1] - pos[1],
                           self.goal_position[0] - pos[0])
        return angle

    def analyze_state(self, state):
        x = state % self.map_size
        y = (state // self.map_size) % self.map_size
        direction = state // (self.map_size * self.map_size)
        direction_names = ['Up', 'Right', 'Down', 'Left']
        print(f"\nState Analysis:")
        print(f"Position: ({x}, {y})")
        print(f"Direction: {direction_names[direction]}")
        print(f"Q-values:")
        print(f"North: {self.q_table[state][0]:.2f}")
        print(f"Right: {self.q_table[state][1]:.2f}")
        print(f"South: {self.q_table[state][2]:.2f}")
        print(f"Left: {self.q_table[state][2]:.2f}")

    def run(self):
        episode = 0
        max_episodes = 100000
        best_episode_steps = float('inf')

        while self.robot.step(self.timestep) != -1 and episode < max_episodes:
            if self.steps_in_episode == 0:
                self.reset_position()
                self.consecutive_back_moves = 0

            self.total_steps += 1
            self.steps_in_episode += 1 

            current_state = self.get_state()
            action = self.get_action(current_state)

            self.take_action(action)
            self.robot.step(self.timestep * 2)

            new_state = self.get_state()
            reward, episode_done = self.get_reward()

            # Update Q-table
            old_value = self.q_table[current_state, action]
            next_max = np.max(self.q_table[new_state])
            new_value = (1 - self.learning_rate) * old_value + \
                        self.learning_rate * (reward + self.discount_factor * next_max)
            self.q_table[current_state, action] = new_value

            # Track changes
            if current_state not in self.episode_changes:
                self.episode_changes[current_state] = []
            self.episode_changes[current_state].append({
                'old': old_value,
                'new': new_value,
                'action': action,
                'x': current_state % self.map_size,
                'y': (current_state // self.map_size) % self.map_size
            })

            if episode_done or self.steps_in_episode >= self.max_steps_per_episode:
                if self.steps_in_episode < best_episode_steps:
                    best_episode_steps = self.steps_in_episode
                    print(f"New best episode steps: {best_episode_steps}")
                print("\nSignificant Q-value changes this episode:")
                for state, changes in self.episode_changes.items():
                    for change in changes:
                        if abs(change['new'] - change['old']) > 0.1:
                            print(f"State {state} (x:{change['x']}, y:{change['y']})")
                            print(f"Action: {['North', 'East', 'South', 'West'][change['action']]}")
                            print(f"Q-value: {change['old']:.2f} -> {change['new']:.2f}")
                            print("---")
                self.episode_changes = {}

                print(f"Episode {episode + 1} completed")
                print(f"Total steps: {self.total_steps}")
                print(f"Steps in episode: {self.steps_in_episode}")
                print(f"Distance to goal: {self.get_distance_to_goal():.2f}")
                print(f"Total collisions: {self.collisions}")
                if self.steps_in_episode < best_episode_steps:
                    best_episode_steps = self.steps_in_episode
                    print(f"New best episode steps: {best_episode_steps}")
                print(f"Best episode steps: {best_episode_steps}")
                print(f"Current epsilon: {self.epsilon:.3f}")
                print("------------------------")

                episode += 1
                self.steps_in_episode = 0
                self.prev_distance_to_goal = None
                if episode % 10 == 0:
                    self.save_q_table()

                self.epsilon = max(0.05, self.epsilon * 0.999)


def main():
    controller = KheperaController()
    controller.run()


if __name__ == "__main__":
    main()