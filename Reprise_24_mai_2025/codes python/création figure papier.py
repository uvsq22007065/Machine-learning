import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

# Define color scheme for different layer types
COLORS = {
    'input': '#E3F2FD',    # Light Blue
    'conv': '#2196F3',     # Blue
    'lstm': '#FF9800',     # Orange
    'dense': '#4CAF50',    # Green
    'dropout': '#F44336',  # Red
    'attention': '#9C27B0' # Purple
}

# Remove the seaborn style line and replace with manual configuration
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.color': '#E6E6E6',
    'grid.linestyle': '-',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.5
})

# Modifier la taille de la figure et l'espacement
fig = plt.figure(figsize=(28, 16))  # Figure encore plus grande
gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.4)  # Plus d'espace entre colonnes

def create_box(ax, x, y, width, height, text, layer_type='dense'):
    box = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.15",
        facecolor=COLORS.get(layer_type, 'white'),
        edgecolor='black',  # Bordure plus foncée
        alpha=1.0,
        linewidth=1.5,  # Bordure plus épaisse
        zorder=3
    )
    ax.add_patch(box)
    lines = text.split('\n')
    total_lines = len(lines)
    for i, line in enumerate(lines):
        y_pos = y + height * (0.7 - (i * 0.4/total_lines))  # Meilleur espacement du texte
        ax.text(x + width/2, y_pos, line, 
                ha='center', va='center', 
                fontsize=12,  # Texte plus grand
                color='black' if layer_type == 'input' else 'white',
                fontweight='bold',
                zorder=4)

def add_arrow(ax, x1, y1, x2, y2):
    ax.arrow(x1, y1, x2-x1, y2-y1, 
             head_width=0.12,
             head_length=0.15,
             fc='black', 
             ec='black', 
             linewidth=2,
             length_includes_head=True,
             zorder=2)  # Flèches en dessous des boîtes

# Standardized layer format
def format_layer(name, size, activation=None, kernel_size=None):
    if kernel_size:
        return f"{name}({size}, {kernel_size})\nAct: {activation}"
    elif activation:
        return f"{name}({size})\nAct: {activation}"
    else:
        return f"{name}({size})"

# Updated models configuration with layer types
models = {
    'CNN-LSTM': [
        ('Input\nSeq.Len = 130', 0, 'input'),
        (format_layer('Conv1D', 128, 'ReLU', 3), 1, 'conv'),
        ('MaxPool1D', 2, 'conv'),
        (format_layer('Conv1D', 64, 'ReLU', 3), 3, 'conv'),
        ('MaxPool1D', 4, 'conv'),
        (format_layer('LSTM', 50, '—'), 5, 'lstm'),
        (format_layer('Dense', 32, 'ReLU'), 6, 'dense'),
        (format_layer('Dense', 1, 'Linear'), 7, 'dense')
    ],
    'Stacked LSTM': [
        ('Input\nSeq.Len = 1', 0, 'input'),
        (format_layer('LSTM', 100, 'ReLU'), 1, 'lstm'),
        (format_layer('LSTM', 50, 'ReLU'), 2, 'lstm'),
        (format_layer('Dense', 1, 'Linear'), 3, 'dense')
    ],
    'RNN (LSTM)': [
        ('Input\nSeq.Len = 1', 0, 'input'),
        (format_layer('LSTM', 128, 'tanh'), 1, 'lstm'),
        (format_layer('LSTM', 64, 'tanh'), 2, 'lstm'),
        (format_layer('Dense', 32, 'ReLU'), 3, 'dense'),
        (format_layer('Dense', 1, 'Linear'), 4, 'dense')
    ],
    'NRAX': [
        ('Input\nSeq.Len = *', 0, 'input'),
        (format_layer('Dense', 64, 'ReLU'), 1, 'dense'),
        ('Self-Attention\n(Self-Attn)', 2, 'attention'),
        ('Concat', 3, 'dense'),
        (format_layer('Dense', 32, 'ReLU'), 4, 'dense'),
        (format_layer('Dense', 1, 'Linear'), 5, 'dense')
    ]
}

# Create hyperparameter box
def add_hyperparameters_box(ax, model_name):
    params_box = patches.FancyBboxPatch(
        (-0.5, 8), 4, 1,
        boxstyle="round,pad=0.3",
        facecolor='white',
        edgecolor='gray',
        alpha=0.95,  # Légèrement transparent
        linewidth=1.0,
        zorder=4  # Au-dessus de tout
    )
    ax.add_patch(params_box)
    
    # Title
    ax.text(1.5, 8.7, f'({chr(65+idx)}) {model_name}', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Parameters
    params_text = 'Adam LR 10⁻⁴\n'
    params_text += 'ES: val_loss; patience 5' if model_name == 'CNN-LSTM' else 'ES: val_loss; patience 10'
    ax.text(1.5, 8.3, params_text, ha='center', va='center', fontsize=14)

# Create legend
def add_legend(fig):
    legend_elements = [
        patches.Patch(facecolor=color, label=layer_type.capitalize())
        for layer_type, color in COLORS.items()
    ]
    fig.legend(handles=legend_elements, 
               loc='lower center', 
               bbox_to_anchor=(0.5, 0.02),
               ncol=len(COLORS),
               fontsize=12)

# Main loop for creating the visualization
for idx, (model_name, layers) in enumerate(models.items()):
    ax = fig.add_subplot(gs[0, idx])
    ax.set_xlim(-1, 3.5)  # Plus d'espace horizontal
    ax.set_ylim(-0.5, 9.5)
    
    # Title and hyperparameters with background
    add_hyperparameters_box(ax, model_name)
    
    # Définir les dropouts pour chaque modèle
    if model_name == 'CNN-LSTM':
        dropouts = [(2.8, 'Dropout p = 0.3'), (4.2, 'Dropout p = 0.3'), (5.6, 'Dropout p = 0.2')]
    elif model_name == 'Stacked LSTM':
        dropouts = [(1.5, 'Dropout p = 0.2')]
    elif model_name == 'RNN (LSTM)':
        dropouts = [(2.5, 'Dropout p = 0.2')]
    elif model_name == 'NRAX':
        dropouts = [(1.5, 'Dropout p = 0.3'), (3.5, 'Dropout p = 0.3')]
    
    # Nouvelles dimensions des boîtes et espacement
    box_width = 1.6  # Boîtes plus larges
    box_height = 0.8  # Boîtes plus hautes
    y_scaling = 1.4  # Plus d'espace vertical
    
    # Ajuster position des boîtes selon le nombre total de couches
    total_layers = len(layers)
    y_offset = (9 - (total_layers * y_scaling)) / 2  # Centrer verticalement
    
    for i, (layer_text, y_pos, layer_type) in enumerate(layers):
        adjusted_y = y_offset + (y_pos * y_scaling)
        create_box(ax, 0, adjusted_y, box_width, box_height, layer_text, layer_type)
        if i < len(layers) - 1:
            next_y = y_offset + (layers[i+1][1] * y_scaling)
            # Ajuster position des flèches
            add_arrow(ax, box_width/2, adjusted_y + box_height/2,
                     box_width/2, next_y + box_height)
    
    # Ajuster position des annotations dropout
    for y_pos, text in dropouts:
        adjusted_y = y_offset + (y_pos * y_scaling)
        dropout_box = patches.Rectangle(
            (1.8, adjusted_y-0.15), 1.4, 0.35,  # Boîte plus grande
            facecolor='white',
            edgecolor='darkgray',  # Bordure plus visible
            alpha=1.0,
            zorder=5,
            linewidth=1.0
        )
        ax.add_patch(dropout_box)
        ax.text(2.5, adjusted_y, text, 
                ha='center', va='center', 
                fontsize=11,  # Texte plus grand
                zorder=6)  # Texte toujours visible

# Bottom text with improved readability
plt.figtext(0.5, 0.02, 
           'Shared training setup: Batch size = 32; max epochs = 1000; loss = MSE\n' +
           'ES: 20% validation split; z-score standardized inputs; MP = MaxPooling1D',
           ha='center', va='center', 
           wrap=True, 
           fontsize=11,  # Texte plus grand
           bbox=dict(facecolor='white', 
                    edgecolor='darkgray',
                    alpha=1.0,
                    pad=8,
                    zorder=7))  # Texte du bas toujours visible

# Adjust layout margins for better spacing
plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.12)

# Add legend
add_legend(fig)

# Show the plot
plt.show()