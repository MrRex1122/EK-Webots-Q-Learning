import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_heatmaps(q_table_path='q_table.npy'):
    q_table = np.load(q_table_path)
    map_size = 12

    # First map - directions with risks
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Create a grid
    for x in range(map_size):
        for y in range(map_size):
            ax1.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False))
            state = x + y * map_size

            # Добавляем маркеры для опасных направлений (Q < -1)
            if q_table[state][0] < -1:  # North
                ax1.plot(x, y + 0.3, 'r.', markersize=10)
            if q_table[state][1] < -1:  # East
                ax1.plot(x + 0.3, y, 'r.', markersize=10)
            if q_table[state][2] < -1:  # South
                ax1.plot(x, y - 0.3, 'r.', markersize=10)
            if q_table[state][3] < -1:  # West
                ax1.plot(x - 0.3, y, 'r.', markersize=10)

    ax1.set_title('Risk Map (red dots show dangerous directions)')
    ax1.set_aspect('equal')
    ax1.set_ylim(ax1.get_ylim()[::-1])
    ax1.invert_yaxis()

    # Second map - best actions
    best_actions = np.zeros((map_size, map_size))
    for x in range(map_size):
        for y in range(map_size):
            state = x + y * map_size
            best_actions[y][x] = np.argmax(q_table[state])

    action_map = sns.heatmap(best_actions, ax=ax2, cmap='coolwarm',
                             cbar_kws={'ticks': [0, 1, 2, 3],
                                       'label': 'Direction',
                                       'format': lambda x, _: ['North', 'East', 'South', 'West'][int(x)]})

    ax2.set_title('Best Actions')

    ax2.invert_xaxis()

    # Add start and finish on both maps
    ax1.plot(1, 10, 'go', label='Goal')
    ax1.plot(10, 1, 'bo', label='Start')
    ax2.plot(1.5, 10.5, 'bo', label='Start')
    ax2.plot(10.5, 1.5, 'go', label='Goal')

    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig('q_table_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save readable version in Excel
    data = []
    for x in range(map_size):
        for y in range(map_size):
            state = x + y * map_size
            row = {
                'X': x,
                'Y': y,
                'North': q_table[state][0],
                'East': q_table[state][1],
                'South': q_table[state][2],
                'West': q_table[state][3],
                'Best_Action': ['North', 'East', 'South', 'West'][int(best_actions[y][x])]
            }
            data.append(row)

    df = pd.DataFrame(data)
    df.to_excel('q_table_readable.xlsx', index=False)


if __name__ == "__main__":
    create_heatmaps()