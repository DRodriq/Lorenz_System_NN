import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statistics
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

class LorenzPredictor:
    def __init__(self):
        self.model = Sequential([
            Dense(3, activation='relu', input_shape=(3,)),
            Dense(8, activation='relu'),
            Dense(3)
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    def prepare_data(self, states):
        X = states[:-1]
        y = states[1:]
        return X, y

    def train(self, states, epochs=100, batch_size=32):
        X, y = self.prepare_data(states)
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2)
        return history

    def predict_next_state(self, current_state):
        return self.model.predict(np.array([current_state]))[0]
    
    def generate_predictions(self, initial_state, num_predictions):
        predictions = [initial_state]
        for _ in range(num_predictions):
            next_state = self.predict_next_state(predictions[-1])
            predictions.append(next_state)
        return np.array(predictions)

class LorenzSystem:
    def __init__(self, sigma=10, rho=28, beta=8/3, dt=0.01):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.results = []

    def step(self, state):
        x, y, z = state
        dx = self.sigma * (y - x) * self.dt
        dy = (x * (self.rho - z) - y) * self.dt
        dz = (x * y - self.beta * z) * self.dt
        return [x + dx, y + dy, z + dz]

    def simulate(self, initial_state, num_steps):
        states = [initial_state]
        for _ in range(num_steps):
            states.append(self.step(states[-1]))
        return np.array(states)

    def plot(self, labels):
        states = self.results
        if len(states) == 0:
            return

        # Separate stable and unstable points
        stable_points = []
        unstable_points = []

        if labels is not None and len(labels) == len(states):
            for state, label in zip(states, labels):
                if label == 'Stable':
                    stable_points.append(state)
                else:
                    unstable_points.append(state)
        else:
            stable_points = states

        # Convert lists to numpy arrays for plotting
        stable_points = np.array(stable_points)
        unstable_points = np.array(unstable_points)

        # Plot stable points
        if stable_points.size > 0:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(stable_points[:, 0], stable_points[:, 1], stable_points[:, 2], color='black')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Stable Points')
            plt.show()

        # Plot unstable points
        if unstable_points.size > 0:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(unstable_points[:, 0], unstable_points[:, 1], unstable_points[:, 2], color='red')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Unstable Points')
            plt.show()

    def run_simulation(self, initial_state, num_steps):
        self.results = self.simulate(initial_state, num_steps)
        return self.results

    def jacobian_eigenvalues(self, point):
        x, y, z = point
        # Define the Jacobian matrix at the given point
        J = np.array([
            [-self.sigma, self.sigma, 0],
            [self.rho - z, -1, -x],
            [y, x, -self.beta]
        ])
        # Calculate the eigenvalues
        eigenvalues = np.linalg.eigvals(J)
        return eigenvalues

def label_values(values, window_size=5, mean=10):
    labels = []
    length = len(values)
    
    for i in range(length):
        # Get the preceding n values
        preceding = values[max(0, i - window_size):i]
        # Get the next n values
        following = values[i + 1:i + 1 + window_size]
        
        # Check if all preceding and following values are less than x
        all_less = all(v < mean for v in preceding + following)
        # Check if all preceding and following values are greater than x
        all_greater = all(v > mean for v in preceding + following)
        
        # Determine the label based on the conditions
        if all_less or all_greater:
            labels.append('Stable')
        else:
            labels.append('Unstable')
    
    return labels

def plot_values_with_mean(values):
    # Calculate the mean of the values
    mean_value = statistics.mean(values)

    # Create a plot
    plt.figure(figsize=(10, 6))
    
    # Plot the values
    plt.plot(values, label='Values', color='blue')
    
    # Plot the mean as a horizontal line
    plt.axhline(y=mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.2f}')
    
    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Values with Mean Line')
    plt.legend()
    
    # Show the plot
    plt.show()

def label_values_distance(distances):
    mean_dis = statistics.mean(distances)
    labels = []
    for dis in distances:
        if dis < mean_dis:
            labels.append('Stable')
        else:
            labels.append('Unstable')
    return labels

def calculate_distances(points):
    distances = []
    for i in range(len(points) - 1):
        # Get the current point and the next point
        point1 = np.array(points[i])
        point2 = np.array(points[i + 1])
        
        # Calculate the Euclidean distance between the two points
        distance = np.linalg.norm(point2 - point1)
        
        # Append the distance to the list
        distances.append(distance)
    
    return distances


# Example usage
if __name__ == "__main__":
    lorenz = LorenzSystem(rho=28)
    initial_state = [1, 1, 1]
    num_steps = 10000
    states = lorenz.run_simulation(initial_state, num_steps)

    #lorenz.plot()

    predictor = LorenzPredictor()
    predictor.model.summary()
    history = predictor.train(states)
    #predictor.plot_training_history(history)


    # Test prediction
    test_state = states[-1]
    predicted_next_state = predictor.predict_next_state(test_state)
    actual_next_state = lorenz.step(test_state)

    print("Predicted next state:", predicted_next_state)
    print("Actual next state:", actual_next_state)

    # Generate 1000 predictions
    num_predictions = 1000
    predictions = predictor.generate_predictions(states[-1], num_predictions)

    # Plot the predictions
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(predictions[:, 0], predictions[:, 1], predictions[:, 2], label='Predictions')
    ax.plot(states[:, 0], states[:, 1], states[:, 2], label='Actual')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.set_title('Lorenz System: Actual vs Predicted')
    ax.legend()
    plt.show()


    '''
    x_values = [sublist[0] for sublist in states]
    x_mean = statistics.mean(x_values)

    distances = calculate_distances(states)

    labels = label_values_distance(distances) + ["Stable"]

    #plot_values_with_mean(x_values)

    window_size = 5
    #labels = label_values(x_values, window_size, x_mean)
    #print(labels)

    print(type(labels))
    print(set(labels))
    print(len(labels))
    print(len([x for x in labels if x == 'Stable']))
    print(len([x for x in labels if x == 'Unstable']))
    lorenz.plot(labels)


    #lorenz.plot()

    predictor = LorenzPredictor()
    predictor.model.summary()
    history = predictor.train(states)
    predictor.plot_training_history(history)


    # Test prediction
    test_state = states[-1]
    predicted_next_state = predictor.predict_next_state(test_state)
    actual_next_state = lorenz.step(test_state)

    print("Predicted next state:", predicted_next_state)
    print("Actual next state:", actual_next_state)

    # Generate 1000 predictions
    num_predictions = 1000
    predictions = predictor.generate_predictions(states[-1], num_predictions)

    # Plot the predictions
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(predictions[:, 0], predictions[:, 1], predictions[:, 2], label='Predictions')
    ax.plot(states[:, 0], states[:, 1], states[:, 2], label='Actual')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Lorenz System: Actual vs Predicted')
    ax.legend()
    plt.show()
    '''



