import pickle
import matplotlib
import matplotlib.pyplot as plt

GRID_COLS = 100
GRID_ROWS = 50

def displayPlotFromPathData():
    all_episode_paths = []
    with open("episode_paths.pkl", 'rb') as fpaths:
        all_episode_paths = pickle.load(fpaths)
    
    episode_count = len(all_episode_paths)

    plt.figure(figsize=(10, 8))
    for i, path in enumerate(all_episode_paths):
        plt.plot(path[:, 0], path[:, 1], label=f'Episode {i+1}')
        
    #plt.xlim(0, GRID_COLS)
    plt.ylim(0, GRID_ROWS)
    plt.title(f"Paths for {episode_count} Episodes")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.show(block = True)

if __name__ == '__main__':
    displayPlotFromPathData()