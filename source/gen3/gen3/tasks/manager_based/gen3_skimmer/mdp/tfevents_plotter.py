#!/usr/bin/env python3
"""
TensorBoard Events File Plotter

This script reads TensorBoard event files (tfevents) and provides an interactive
interface to plot training data from RSL-RL PPO training.

Usage:
    python3 tfevents_plotter.py --events-file /path/to/events.out.tfevents.xxx

FONT SIZE CUSTOMIZATION:
    To change font sizes, edit the FONT_SIZES dictionary below:
    
    - 'title': Size of plot titles (default: 28)
    - 'axis_label': Size of X/Y axis labels (default: 28) 
    - 'axis_tick': Size of axis tick numbers (default: 36)
    - 'legend': Size of legend text (default: 28)
    - 'figure_size': Base figure size (default: 10)
    
    Quick presets available:
    - FONT_PRESETS['small']: Compact plots
    - FONT_PRESETS['medium']: Standard size
    - FONT_PRESETS['large']: Large fonts
    - FONT_PRESETS['extra_large']: Very large fonts
    
    You can also change fonts interactively during runtime using option 7.

LINE THICKNESS CUSTOMIZATION:
    To change line thicknesses, edit the LINE_THICKNESS dictionary below:
    
    - 'raw_data': Thickness of raw data lines (default: 0.5)
    - 'smoothed_data': Thickness of smoothed data lines (default: 2.0)
    - 'histogram': Thickness of histogram lines (default: 1.5)
    
    Quick presets available:
    - LINE_THICKNESS_PRESETS['thin']: Thin lines
    - LINE_THICKNESS_PRESETS['medium']: Standard thickness
    - LINE_THICKNESS_PRESETS['thick']: Thick lines
    - LINE_THICKNESS_PRESETS['extra_thick']: Very thick lines
    
    You can also change line thicknesses interactively during runtime using option 8.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import re

# =============================================================================
# FONT SIZE CONFIGURATION - Easy to adjust all font sizes here
# =============================================================================
FONT_SIZES = {
    'title': 28,           # Plot titles (bold) - Increase for bigger titles
    'axis_label': 28,      # X and Y axis labels - Increase for bigger labels
    'axis_tick': 36,       # Axis tick numbers - Increase for bigger numbers
    'legend': 28,          # Legend text - Increase for bigger legend
    'figure_size': 10,     # Base figure size - Increase for bigger plots overall
}

# Additional font size presets for different use cases:
FONT_PRESETS = {
    'small': {
        'title': 14, 'axis_label': 12, 'axis_tick': 10, 'legend': 10, 'figure_size': 6
    },
    'medium': {
        'title': 18, 'axis_label': 16, 'axis_tick': 14, 'legend': 14, 'figure_size': 8
    },
    'large': {
        'title': 24, 'axis_label': 22, 'axis_tick': 20, 'legend': 20, 'figure_size': 12
    },
    'extra_large': {
        'title': 28, 'axis_label': 26, 'axis_tick': 24, 'legend': 24, 'figure_size': 14
    }
}

# To use a preset, uncomment one of these lines:
# FONT_SIZES.update(FONT_PRESETS['large'])        # For large fonts
# FONT_SIZES.update(FONT_PRESETS['extra_large'])  # For extra large fonts

# =============================================================================
# LINE THICKNESS CONFIGURATION - Easy to adjust line thicknesses here
# =============================================================================
LINE_THICKNESS = {
    'raw_data': 0.5,       # Raw data lines (pale, transparent) - Increase for thicker raw lines
    'smoothed_data': 4.0,  # Smoothed data lines (solid color) - Increase for thicker smoothed lines
    'histogram': 1.5,      # Histogram lines - Increase for thicker histogram lines
}

# Additional line thickness presets for different use cases:
LINE_THICKNESS_PRESETS = {
    'thin': {
        'raw_data': 0.3, 'smoothed_data': 1.0, 'histogram': 1.0
    },
    'medium': {
        'raw_data': 0.5, 'smoothed_data': 2.0, 'histogram': 1.5
    },
    'thick': {
        'raw_data': 0.8, 'smoothed_data': 3.0, 'histogram': 2.5
    },
    'extra_thick': {
        'raw_data': 1.2, 'smoothed_data': 4.0, 'histogram': 3.5
    }
}

# To use a preset, uncomment one of these lines:
# LINE_THICKNESS.update(LINE_THICKNESS_PRESETS['thick'])        # For thick lines
# LINE_THICKNESS.update(LINE_THICKNESS_PRESETS['extra_thick'])  # For extra thick lines

def change_line_thickness(preset_name=None, custom_thickness=None):
    """
    Change line thicknesses dynamically.
    
    Args:
        preset_name: Name of preset ('thin', 'medium', 'thick', 'extra_thick')
        custom_thickness: Dictionary with custom line thicknesses
    """
    global LINE_THICKNESS
    
    if preset_name and preset_name in LINE_THICKNESS_PRESETS:
        LINE_THICKNESS.update(LINE_THICKNESS_PRESETS[preset_name])
        print(f"Applied line thickness preset: {preset_name}")
        print(f"New line thicknesses: {LINE_THICKNESS}")
    elif custom_thickness:
        LINE_THICKNESS.update(custom_thickness)
        print("Applied custom line thicknesses")
        print(f"New line thicknesses: {LINE_THICKNESS}")
    else:
        print("Available presets:", list(LINE_THICKNESS_PRESETS.keys()))
        print("Current line thicknesses:", LINE_THICKNESS)

# =============================================================================
# END LINE THICKNESS CONFIGURATION
# =============================================================================

def change_font_sizes(preset_name=None, custom_sizes=None):
    """
    Change font sizes dynamically.
    
    Args:
        preset_name: Name of preset ('small', 'medium', 'large', 'extra_large')
        custom_sizes: Dictionary with custom font sizes
    """
    global FONT_SIZES
    
    if preset_name and preset_name in FONT_PRESETS:
        FONT_SIZES.update(FONT_PRESETS[preset_name])
        print(f"Applied font preset: {preset_name}")
        print(f"New font sizes: {FONT_SIZES}")
    elif custom_sizes:
        FONT_SIZES.update(custom_sizes)
        print("Applied custom font sizes")
        print(f"New font sizes: {FONT_SIZES}")
    else:
        print("Available presets:", list(FONT_PRESETS.keys()))
        print("Current font sizes:", FONT_SIZES)

# =============================================================================
# END FONT SIZE CONFIGURATION
# =============================================================================

def load_tfevents_data(events_file):
    """
    Load data from TensorBoard event file.
    
    Args:
        events_file: Path to the TensorBoard event file
        
    Returns:
        Dictionary containing all available data
    """
    print(f"Loading TensorBoard events from: {events_file}")
    
    # Create EventAccumulator
    ea = EventAccumulator(events_file)
    ea.Reload()
    
    # Get all available tags
    tags = ea.Tags()
    print(f"Available tags: {tags}")
    
    data = {}
    
    # Extract scalar data
    if 'scalars' in tags:
        for tag in tags['scalars']:
            print(f"Loading scalar data for tag: {tag}")
            events = ea.Scalars(tag)
            
            # Extract step and value data
            steps = [event.step for event in events]
            values = [event.value for event in events]
            
            data[tag] = {
                'steps': steps,
                'values': values,
                'type': 'scalar'
            }
    
    # Extract histogram data
    if 'histograms' in tags:
        for tag in tags['histograms']:
            print(f"Loading histogram data for tag: {tag}")
            events = ea.Histograms(tag)
            
            # Extract step and histogram data
            steps = [event.step for event in events]
            histograms = [event.histogram_value for event in events]
            
            data[tag] = {
                'steps': steps,
                'values': histograms,
                'type': 'histogram'
            }
    
    # Extract image data
    if 'images' in tags:
        for tag in tags['images']:
            print(f"Loading image data for tag: {tag}")
            events = ea.Images(tag)
            
            # Extract step and image data
            steps = [event.step for event in events]
            images = [event.encoded_image_string for event in events]
            
            data[tag] = {
                'steps': steps,
                'values': images,
                'type': 'image'
            }
    
    print(f"Loaded {len(data)} data series")
    return data

def smooth_data(values, smoothing_factor=0.6):
    """
    Apply exponential moving average smoothing to data.
    
    Args:
        values: List of values to smooth
        smoothing_factor: Smoothing coefficient (0-1), higher = more smoothing
        
    Returns:
        Smoothed values
    """
    if not values:
        return values
    
    smoothed = [values[0]]
    for i in range(1, len(values)):
        smoothed_value = smoothing_factor * smoothed[i-1] + (1 - smoothing_factor) * values[i]
        smoothed.append(smoothed_value)
    
    return smoothed

def plot_scalar_data(tag, data_list, folder_names, save_path=None, show_plot=True, smoothing_factor=0.6):
    """
    Plot scalar data from multiple files with smoothing.
    
    Args:
        tag: Tag name for the data
        data_list: List of dictionaries containing step and value data
        folder_names: List of folder names for legend
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
        smoothing_factor: Smoothing coefficient (0-1), higher = more smoothing
    """
    colors = ['red', 'green', 'blue']
    
    # Use configurable figure size
    fig_size = FONT_SIZES['figure_size']
    plt.figure(figsize=(fig_size, fig_size))
    
    for i, (data, folder_name) in enumerate(zip(data_list, folder_names)):
        if i < len(colors):
            color = colors[i]
        else:
            color = f'C{i}'  # Use matplotlib's default color cycle
        
        steps = data['steps']
        values = data['values']
        
        # Plot raw data (pale and transparent), no legend entry
        plt.plot(steps, values, color=color, alpha=0.2, linewidth=LINE_THICKNESS['raw_data'])
        
        # Plot smoothed data (solid color) with formatted legend label
        smoothed_values = smooth_data(values, smoothing_factor)
        label = format_legend_label(folder_name, i)
        plt.plot(steps, smoothed_values, color=color, alpha=0.8, linewidth=LINE_THICKNESS['smoothed_data'], label=label)
    
    plt.title(f'{tag}', fontsize=FONT_SIZES['title'], fontweight='bold')
    plt.xlabel('Training Step', fontsize=FONT_SIZES['axis_label'])
    plt.ylabel('Value', fontsize=FONT_SIZES['axis_label'])
    plt.xticks(fontsize=FONT_SIZES['axis_tick'])
    plt.yticks(fontsize=FONT_SIZES['axis_tick'])
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=FONT_SIZES['legend'])
    plt.tight_layout()
    
    if save_path:
        # Use plot title as filename, clean it for filesystem
        clean_title = re.sub(r'[^\w\-_\.]', '_', tag)
        pdf_path = f"{clean_title}.pdf"
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Plot saved to: {pdf_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()

def plot_histogram_data(tag, data_list, folder_names, save_path=None, show_plot=True, smoothing_factor=0.6):
    """
    Plot histogram data from multiple files with smoothing.
    
    Args:
        tag: Tag name for the data
        data_list: List of dictionaries containing step and histogram data
        folder_names: List of folder names for legend
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
        smoothing_factor: Smoothing coefficient (0-1), higher = more smoothing
    """
    colors = ['red', 'green', 'blue']
    
    # Create a figure with subplots using configurable sizes
    fig_width = FONT_SIZES['figure_size'] * 1.2  # Wider for 2 subplots
    fig_height = FONT_SIZES['figure_size'] * 0.6  # Height for subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    
    # Plot histogram evolution over time (use first file only for clarity)
    if data_list:
        steps = data_list[0]['steps']
        histograms = data_list[0]['values']
        
        for i, (step, hist) in enumerate(zip(steps, histograms)):
            if i % max(1, len(steps) // 10) == 0:  # Plot every 10th histogram
                bins = hist.bucket_limit
                counts = hist.bucket
                ax1.hist(bins[:-1], bins=bins, weights=counts, alpha=0.3, 
                        label=f'Step {step}', linewidth=LINE_THICKNESS['histogram'])
    
    ax1.set_title(f'{tag} - Histogram Evolution', fontsize=FONT_SIZES['title'], fontweight='bold')
    ax1.set_xlabel('Value', fontsize=FONT_SIZES['axis_label'])
    ax1.set_ylabel('Count', fontsize=FONT_SIZES['axis_label'])
    ax1.tick_params(axis='both', which='major', labelsize=FONT_SIZES['axis_tick'])
    ax1.legend(fontsize=FONT_SIZES['legend'])
    ax1.grid(True, alpha=0.3)
    
    # Plot statistics over time for all files with smoothing
    for i, (data, folder_name) in enumerate(zip(data_list, folder_names)):
        if i < len(colors):
            color = colors[i]
        else:
            color = f'C{i}'
        
        steps = data['steps']
        histograms = data['values']
        
        means = []
        stds = []
        for hist in histograms:
            bins = hist.bucket_limit
            counts = hist.bucket
            # Calculate weighted mean and std
            bin_centers = (bins[:-1] + bins[1:]) / 2
            mean = np.average(bin_centers, weights=counts)
            variance = np.average((bin_centers - mean) ** 2, weights=counts)
            std = np.sqrt(variance)
            means.append(mean)
            stds.append(std)
        
        # Plot raw data (pale and transparent)
        ax2.plot(steps, means, color=color, alpha=0.2, linewidth=LINE_THICKNESS['raw_data'])
        
        # Plot smoothed data (solid color)
        smoothed_means = smooth_data(means, smoothing_factor)
        label = format_legend_label(folder_name, i)
        ax2.plot(steps, smoothed_means, color=color, alpha=0.8, linewidth=LINE_THICKNESS['smoothed_data'], label=label)
    
    ax2.set_title(f'{tag} - Statistics Over Time', fontsize=FONT_SIZES['title'], fontweight='bold')
    ax2.set_xlabel('Training Step', fontsize=FONT_SIZES['axis_label'])
    ax2.set_ylabel('Value', fontsize=FONT_SIZES['axis_label'])
    ax2.tick_params(axis='both', which='major', labelsize=FONT_SIZES['axis_tick'])
    ax2.legend(fontsize=FONT_SIZES['legend'])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        # Use plot title as filename, clean it for filesystem
        clean_title = re.sub(r'[^\w\-_\.]', '_', tag)
        pdf_path = f"{clean_title}.pdf"
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Plot saved to: {pdf_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()

def plot_image_data(tag, data_list, folder_names, save_path=None, show_plot=True, smoothing_factor=0.6):
    """
    Plot image data from multiple files.
    
    Args:
        tag: Tag name for the data
        data_list: List of dictionaries containing step and image data
        folder_names: List of folder names for legend
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
        smoothing_factor: Smoothing coefficient (0-1), higher = more smoothing (not used for images)
    """
    for i, (data, folder_name) in enumerate(zip(data_list, folder_names)):
        steps = data['steps']
        images = data['values']
        
        print(f"Found {len(images)} images for tag '{tag}' in {folder_name}")
        print(f"Steps: {steps}")
    
    # You could add image plotting here if needed
    # import io
    # from PIL import Image
    # for i, (step, img_data) in enumerate(zip(steps, images)):
    #     img = Image.open(io.BytesIO(img_data))
    #     plt.figure()
    #     plt.imshow(img)
    #     plt.title(f'{tag} - Step {step}')
    #     plt.show()

def categorize_tags(tags):
    """
    Categorize tags into meaningful groups.
    
    Args:
        tags: List of tag names
        
    Returns:
        Dictionary with categorized and sorted tags
    """
    categories = {
        'Rewards': [],
        'Errors': [],
        'Penalties': [],
        'Actions': [],
        'States': [],
        'Training': [],
        'Other': []
    }
    
    for tag in tags:
        tag_lower = tag.lower()
        
        if any(keyword in tag_lower for keyword in ['reward', 'return', 'value']):
            categories['Rewards'].append(tag)
        elif any(keyword in tag_lower for keyword in ['error', 'loss', 'distance']):
            categories['Errors'].append(tag)
        elif any(keyword in tag_lower for keyword in ['penalty', 'collision']):
            categories['Penalties'].append(tag)
        elif any(keyword in tag_lower for keyword in ['action', 'control', 'command']):
            categories['Actions'].append(tag)
        elif any(keyword in tag_lower for keyword in ['state', 'position', 'orientation', 'joint']):
            categories['States'].append(tag)
        elif any(keyword in tag_lower for keyword in ['train', 'learn', 'gradient', 'lr']):
            categories['Training'].append(tag)
        else:
            categories['Other'].append(tag)
    
    # Sort tags in each category for stable ordering
    for key in categories:
        categories[key] = sorted(categories[key])
    
    return categories


def format_legend_label(folder_name: str, index_zero_based: int) -> str:
    """
    Build legend label as: 'NN - w_p = X, w_o = Y'
    - NN: number identifying the run (parsed from folder name; fallback to index+1)
    - w_p: value after 'pos' (can be negative)
    - w_o: value after 'ori' (can be negative)
    """
    # Extract numeric id (first integer found), fallback to index+1
    m_id = re.search(r'(^|[^0-9])(\d{1,3})(?![0-9])', folder_name)
    try:
        run_id = int(m_id.group(2)) if m_id else (index_zero_based + 1)
    except Exception:
        run_id = index_zero_based + 1
    id_str = f"{run_id:02d}"

    # Extract w_p after 'pos' and w_o after 'ori'
    m_wp = re.search(r'pos_([+-]?[0-9]*\.?[0-9]+)', folder_name)
    m_wo = re.search(r'ori_([+-]?[0-9]*\.?[0-9]+)', folder_name)
    w_p = m_wp.group(1) if m_wp else "?"
    w_o = m_wo.group(1) if m_wo else "?"

    return f"{id_str} - w_p = {w_p}, w_o = {w_o}"

def interactive_plot_interface(all_data, folder_names):
    """
    Provide interactive interface for plotting data from multiple files.
    
    Args:
        all_data: List of dictionaries containing all available data from each file
        folder_names: List of folder names for legend
    """
    print("\n" + "="*60)
    print("TensorBoard Events Plotter - Interactive Interface")
    print("="*60)
    
    # Get all unique tags across all files
    all_tags = set()
    for data in all_data:
        all_tags.update(data.keys())
    
    # Categorize tags
    categories = categorize_tags(list(all_tags))
    
    # Show available categories
    print("\nAvailable data categories:")
    for category, tags in categories.items():
        if tags:
            print(f"\n{category}:")
            for tag in tags:
                # Count how many files have this tag
                tag_count = sum(1 for data in all_data if tag in data)
                print(f"  - {tag} (available in {tag_count}/{len(all_data)} files)")
    
    # Stable list of all tags for consistent indexing
    all_tags = sorted(list(all_tags))
    
    # Default smoothing factor
    smoothing_factor = 0.6
    
    # Interactive selection
    while True:
        print("\n" + "-"*40)
        print("Options:")
        print("1. Plot all data")
        print("2. Plot by category")
        print("3. Plot specific tag")
        print("4. Plot multiple tags")
        print("5. Save all plots")
        print("6. Change smoothing factor")
        print("7. Change font sizes")
        print("8. Change line thickness")
        print("9. Exit")
        
        choice = input("\nEnter your choice (1-9): ").strip()
        
        if choice == '1':
            # Plot all data
            print("\nPlotting all available data...")
            for tag in all_tags:
                # Get data for this tag from all files
                tag_data_list = []
                for data in all_data:
                    if tag in data:
                        tag_data_list.append(data[tag])
                
                if tag_data_list:
                    print(f"\nPlotting: {tag}")
                    tag_data = tag_data_list[0]
                    if tag_data['type'] == 'scalar':
                        plot_scalar_data(tag, tag_data_list, folder_names, show_plot=True, smoothing_factor=smoothing_factor)
                    elif tag_data['type'] == 'histogram':
                        plot_histogram_data(tag, tag_data_list, folder_names, show_plot=True, smoothing_factor=smoothing_factor)
                    elif tag_data['type'] == 'image':
                        plot_image_data(tag, tag_data_list, folder_names, show_plot=True, smoothing_factor=smoothing_factor)
        
        elif choice == '2':
            # Plot by category
            print("\nAvailable categories:")
            for i, (category, tags) in enumerate(categories.items(), 1):
                if tags:
                    print(f"{i}. {category} ({len(tags)} tags)")
            
            try:
                cat_choice = int(input("\nEnter category number: ")) - 1
                category = list(categories.keys())[cat_choice]
                tags = categories[category]
                
                if tags:
                    print(f"\nPlotting {category} data...")
                    for tag in tags:
                        # Get data for this tag from all files
                        tag_data_list = []
                        for data in all_data:
                            if tag in data:
                                tag_data_list.append(data[tag])
                        
                        if tag_data_list:
                            print(f"\nPlotting: {tag}")
                            tag_data = tag_data_list[0]
                            if tag_data['type'] == 'scalar':
                                plot_scalar_data(tag, tag_data_list, folder_names, show_plot=True, smoothing_factor=smoothing_factor)
                            elif tag_data['type'] == 'histogram':
                                plot_histogram_data(tag, tag_data_list, folder_names, show_plot=True, smoothing_factor=smoothing_factor)
                            elif tag_data['type'] == 'image':
                                plot_image_data(tag, tag_data_list, folder_names, show_plot=True, smoothing_factor=smoothing_factor)
                else:
                    print("No tags in this category.")
            except (ValueError, IndexError):
                print("Invalid choice.")
        
        elif choice == '3':
            # Plot specific tag
            print("\nAvailable tags:")
            for i, tag in enumerate(all_tags, 1):
                # Count how many files have this tag
                tag_count = sum(1 for data in all_data if tag in data)
                tag_data = None
                for data in all_data:
                    if tag in data:
                        tag_data = data[tag]
                        break
                
                if tag_data:
                    data_type = tag_data['type']
                    print(f"{i}. {tag} ({data_type}, available in {tag_count}/{len(all_data)} files)")
            
            try:
                tag_choice = int(input("\nEnter tag number: ")) - 1
                tag = list(all_tags)[tag_choice]
                
                # Get data for this tag from all files
                tag_data_list = []
                for data in all_data:
                    if tag in data:
                        tag_data_list.append(data[tag])
                
                if tag_data_list:
                    print(f"\nPlotting: {tag}")
                    tag_data = tag_data_list[0]
                    if tag_data['type'] == 'scalar':
                        plot_scalar_data(tag, tag_data_list, folder_names, show_plot=True, smoothing_factor=smoothing_factor)
                    elif tag_data['type'] == 'histogram':
                        plot_histogram_data(tag, tag_data_list, folder_names, show_plot=True, smoothing_factor=smoothing_factor)
                    elif tag_data['type'] == 'image':
                        plot_image_data(tag, tag_data_list, folder_names, show_plot=True, smoothing_factor=smoothing_factor)
            except (ValueError, IndexError):
                print("Invalid choice.")
        
        elif choice == '4':
            # Plot multiple tags
            print("\nAvailable tags:")
            for i, tag in enumerate(all_tags, 1):
                # Count how many files have this tag
                tag_count = sum(1 for data in all_data if tag in data)
                tag_data = None
                for data in all_data:
                    if tag in data:
                        tag_data = data[tag]
                        break
                
                if tag_data:
                    data_type = tag_data['type']
                    print(f"{i}. {tag} ({data_type}, available in {tag_count}/{len(all_data)} files)")
            
            try:
                tag_input = input("\nEnter tag numbers (comma-separated, or use dashes like 2-5): ").strip()
                
                # Parse input for ranges and individual numbers
                tag_indices = []
                for part in tag_input.split(','):
                    part = part.strip()
                    if '-' in part:
                        # Handle range like "2-5"
                        try:
                            start, end = part.split('-')
                            start_idx = int(start.strip()) - 1
                            end_idx = int(end.strip()) - 1
                            tag_indices.extend(range(start_idx, end_idx + 1))
                        except ValueError:
                            print(f"Invalid range format: {part}")
                            continue
                    else:
                        # Handle individual number
                        try:
                            tag_indices.append(int(part) - 1)
                        except ValueError:
                            print(f"Invalid number: {part}")
                            continue
                
                selected_tags = [list(all_tags)[i] for i in tag_indices if 0 <= i < len(all_tags)]
                
                if selected_tags:
                    print(f"\nPlotting multiple tags: {selected_tags}")
                    
                    # Create subplot for scalar data
                    scalar_tags = []
                    for tag in selected_tags:
                        tag_data = None
                        for data in all_data:
                            if tag in data:
                                tag_data = data[tag]
                                break
                        if tag_data and tag_data['type'] == 'scalar':
                            scalar_tags.append(tag)
                    
                    # Create separate windows for each scalar tag
                    figures = []  # Store all figures to show them at once
                    
                    for tag in scalar_tags:
                        # Get data for this tag from all files
                        tag_data_list = []
                        for data in all_data:
                            if tag in data:
                                tag_data_list.append(data[tag])
                        
                        if tag_data_list:
                            # Create new figure for this tag using configurable size
                            fig_size = FONT_SIZES['figure_size']
                            fig = plt.figure(figsize=(fig_size, fig_size))
                            figures.append(fig)
                            
                            colors = ['red', 'green', 'blue']
                            
                            for j, (tag_data, folder_name) in enumerate(zip(tag_data_list, folder_names)):
                                if j < len(colors):
                                    color = colors[j]
                                else:
                                    color = f'C{j}'
                                
                                # Plot raw data (pale and transparent) without legend
                                plt.plot(tag_data['steps'], tag_data['values'], color=color, alpha=0.2, linewidth=LINE_THICKNESS['raw_data'])
                                
                                # Plot smoothed data (solid color) with formatted legend label
                                smoothed_values = smooth_data(tag_data['values'], smoothing_factor)
                                label = format_legend_label(folder_name, j)
                                plt.plot(tag_data['steps'], smoothed_values, color=color, alpha=0.8, linewidth=LINE_THICKNESS['smoothed_data'], label=label)
                            
                            plt.title(f'{tag}', fontsize=FONT_SIZES['title'], fontweight='bold')
                            plt.xlabel('Training Step', fontsize=FONT_SIZES['axis_label'])
                            plt.ylabel('Value', fontsize=FONT_SIZES['axis_label'])
                            plt.xticks(fontsize=FONT_SIZES['axis_tick'])
                            plt.yticks(fontsize=FONT_SIZES['axis_tick'])
                            plt.grid(True, alpha=0.3)
                            plt.legend(fontsize=FONT_SIZES['legend'])
                            plt.tight_layout()
                    
                    # Show all figures at once
                    if figures:
                        plt.show()
                        # Close all figures after showing
                        for fig in figures:
                            plt.close(fig)
                    
                    # Plot individual histograms and images
                    for tag in selected_tags:
                        tag_data_list = []
                        for data in all_data:
                            if tag in data:
                                tag_data_list.append(data[tag])
                        
                        if tag_data_list:
                            tag_data = tag_data_list[0]
                            if tag_data['type'] == 'histogram':
                                print(f"\nPlotting histogram: {tag}")
                                plot_histogram_data(tag, tag_data_list, folder_names, show_plot=True, smoothing_factor=smoothing_factor)
                            elif tag_data['type'] == 'image':
                                print(f"\nPlotting image: {tag}")
                                plot_image_data(tag, tag_data_list, folder_names, show_plot=True, smoothing_factor=smoothing_factor)
                else:
                    print("No valid tags selected.")
            except (ValueError, IndexError):
                print("Invalid choice.")
        
        elif choice == '5':
            # Save all plots
            save_dir = input("\nEnter directory to save plots (or press Enter for current directory): ").strip()
            if not save_dir:
                save_dir = "."
            
            os.makedirs(save_dir, exist_ok=True)
            print(f"\nSaving all plots to: {save_dir}")
            
            for tag in all_tags:
                # Get data for this tag from all files
                tag_data_list = []
                for data in all_data:
                    if tag in data:
                        tag_data_list.append(data[tag])
                
                if tag_data_list:
                    # Clean tag name for filename
                    clean_tag = re.sub(r'[^\w\-_\.]', '_', tag)
                    save_path = os.path.join(save_dir, f"{clean_tag}.png")
                    
                    print(f"Saving: {tag}")
                    tag_data = tag_data_list[0]
                    if tag_data['type'] == 'scalar':
                        plot_scalar_data(tag, tag_data_list, folder_names, save_path=save_path, show_plot=False, smoothing_factor=smoothing_factor)
                    elif tag_data['type'] == 'histogram':
                        plot_histogram_data(tag, tag_data_list, folder_names, save_path=save_path, show_plot=False, smoothing_factor=smoothing_factor)
                    elif tag_data['type'] == 'image':
                        plot_image_data(tag, tag_data_list, folder_names, save_path=save_path, show_plot=False, smoothing_factor=smoothing_factor)
            
            print(f"\nAll plots saved to: {save_dir}")
        
        elif choice == '6':
            # Change smoothing factor
            try:
                smoothing_input = input(f"\nEnter new smoothing factor (0.0-1.0, current={smoothing_factor:.2f}): ").strip()
                if smoothing_input:
                    smoothing_factor = float(smoothing_input)
                    smoothing_factor = max(0.0, min(1.0, smoothing_factor))  # Clamp to [0, 1]
                print(f"Updated smoothing factor: {smoothing_factor:.2f}")
            except ValueError:
                print(f"Invalid input, keeping current smoothing factor: {smoothing_factor:.2f}")
        
        elif choice == '7':
            # Change font sizes
            print("\nFont Size Options:")
            print("1. Use preset (small/medium/large/extra_large)")
            print("2. Custom font sizes")
            print("3. Show current font sizes")
            print("4. Back to main menu")
            
            font_choice = input("\nEnter your choice (1-4): ").strip()
            
            if font_choice == '1':
                print("\nAvailable presets:")
                for i, preset in enumerate(FONT_PRESETS.keys(), 1):
                    print(f"{i}. {preset}")
                
                try:
                    preset_idx = int(input("\nEnter preset number: ")) - 1
                    preset_names = list(FONT_PRESETS.keys())
                    if 0 <= preset_idx < len(preset_names):
                        preset_name = preset_names[preset_idx]
                        change_font_sizes(preset_name=preset_name)
                    else:
                        print("Invalid preset number.")
                except ValueError:
                    print("Invalid input.")
            
            elif font_choice == '2':
                print("\nEnter custom font sizes (press Enter to keep current):")
                try:
                    title = input(f"Title font size (current: {FONT_SIZES['title']}): ").strip()
                    axis_label = input(f"Axis label font size (current: {FONT_SIZES['axis_label']}): ").strip()
                    axis_tick = input(f"Axis tick font size (current: {FONT_SIZES['axis_tick']}): ").strip()
                    legend = input(f"Legend font size (current: {FONT_SIZES['legend']}): ").strip()
                    figure_size = input(f"Figure size (current: {FONT_SIZES['figure_size']}): ").strip()
                    
                    custom_sizes = {}
                    if title: custom_sizes['title'] = int(title)
                    if axis_label: custom_sizes['axis_label'] = int(axis_label)
                    if axis_tick: custom_sizes['axis_tick'] = int(axis_tick)
                    if legend: custom_sizes['legend'] = int(legend)
                    if figure_size: custom_sizes['figure_size'] = int(figure_size)
                    
                    if custom_sizes:
                        change_font_sizes(custom_sizes=custom_sizes)
                    else:
                        print("No changes made.")
                except ValueError:
                    print("Invalid input. Please enter numbers only.")
            
            elif font_choice == '3':
                print("\nCurrent font sizes:")
                for key, value in FONT_SIZES.items():
                    print(f"  {key}: {value}")
            
            elif font_choice == '4':
                continue
            else:
                print("Invalid choice.")
        
        elif choice == '8':
            # Change line thickness
            print("\nLine Thickness Options:")
            print("1. Use preset (thin/medium/thick/extra_thick)")
            print("2. Custom line thicknesses")
            print("3. Show current line thicknesses")
            print("4. Back to main menu")
            
            thickness_choice = input("\nEnter your choice (1-4): ").strip()
            
            if thickness_choice == '1':
                print("\nAvailable presets:")
                for i, preset in enumerate(LINE_THICKNESS_PRESETS.keys(), 1):
                    print(f"{i}. {preset}")
                
                try:
                    preset_idx = int(input("\nEnter preset number: ")) - 1
                    preset_names = list(LINE_THICKNESS_PRESETS.keys())
                    if 0 <= preset_idx < len(preset_names):
                        preset_name = preset_names[preset_idx]
                        change_line_thickness(preset_name=preset_name)
                    else:
                        print("Invalid preset number.")
                except ValueError:
                    print("Invalid input.")
            
            elif thickness_choice == '2':
                print("\nEnter custom line thicknesses (press Enter to keep current):")
                try:
                    raw_data = input(f"Raw data line thickness (current: {LINE_THICKNESS['raw_data']}): ").strip()
                    smoothed_data = input(f"Smoothed data line thickness (current: {LINE_THICKNESS['smoothed_data']}): ").strip()
                    histogram = input(f"Histogram line thickness (current: {LINE_THICKNESS['histogram']}): ").strip()
                    
                    custom_thickness = {}
                    if raw_data: custom_thickness['raw_data'] = float(raw_data)
                    if smoothed_data: custom_thickness['smoothed_data'] = float(smoothed_data)
                    if histogram: custom_thickness['histogram'] = float(histogram)
                    
                    if custom_thickness:
                        change_line_thickness(custom_thickness=custom_thickness)
                    else:
                        print("No changes made.")
                except ValueError:
                    print("Invalid input. Please enter numbers only.")
            
            elif thickness_choice == '3':
                print("\nCurrent line thicknesses:")
                for key, value in LINE_THICKNESS.items():
                    print(f"  {key}: {value}")
            
            elif thickness_choice == '4':
                continue
            else:
                print("Invalid choice.")
        
        elif choice == '9':
            print("\nExiting...")
            break
        
        else:
            print("Invalid choice. Please enter a number between 1 and 9.")

def main():
    """Main function to process TensorBoard event files."""
    
    parser = argparse.ArgumentParser(description='TensorBoard Events File Plotter')
    parser.add_argument('events_files', type=str, nargs='+',
                       help='Paths to TensorBoard event files (up to 3 files)')
    parser.add_argument('--auto-plot', action='store_true',
                       help='Automatically plot all data without interactive interface')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save plots (only used with --auto-plot)')
    
    args = parser.parse_args()
    
    # Limit to 3 files
    if len(args.events_files) > 3:
        print("Warning: Only the first 3 files will be processed.")
        args.events_files = args.events_files[:3]
    
    # Check if files exist and load data
    all_data = []
    folder_names = []
    
    for events_file in args.events_files:
        if not os.path.exists(events_file):
            print(f"Error: File not found: {events_file}")
            continue
        
        # Load data from TensorBoard event file
        data = load_tfevents_data(events_file)
        
        if data:
            all_data.append(data)
            # Extract folder name for legend
            folder_name = os.path.basename(os.path.dirname(events_file))
            if not folder_name:
                folder_name = os.path.basename(events_file)
            folder_names.append(folder_name)
    
    if not all_data:
        print("No data found in any of the event files.")
        return
    
    print(f"\nLoaded data from {len(all_data)} files:")
    for i, folder_name in enumerate(folder_names):
        print(f"  {i+1}. {folder_name}")
    
    if args.auto_plot:
        # Auto-plot mode
        smoothing_factor = 0.6  # Default smoothing factor for auto-plot
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            print(f"Auto-plotting all data to: {args.save_dir}")
            
            # Get all unique tags across all files
            all_tags = set()
            for data in all_data:
                all_tags.update(data.keys())
            
            for tag in all_tags:
                # Get data for this tag from all files
                tag_data_list = []
                for data in all_data:
                    if tag in data:
                        tag_data_list.append(data[tag])
                
                if tag_data_list:
                    clean_tag = re.sub(r'[^\w\-_\.]', '_', tag)
                    save_path = os.path.join(args.save_dir, f"{clean_tag}.png")
                    
                    print(f"Plotting: {tag}")
                    tag_data = tag_data_list[0]
                    if tag_data['type'] == 'scalar':
                        plot_scalar_data(tag, tag_data_list, folder_names, save_path=save_path, show_plot=False, smoothing_factor=smoothing_factor)
                    elif tag_data['type'] == 'histogram':
                        plot_histogram_data(tag, tag_data_list, folder_names, save_path=save_path, show_plot=False, smoothing_factor=smoothing_factor)
                    elif tag_data['type'] == 'image':
                        plot_image_data(tag, tag_data_list, folder_names, save_path=save_path, show_plot=False, smoothing_factor=smoothing_factor)
            
            print(f"All plots saved to: {args.save_dir}")
        else:
            print("Auto-plotting all data...")
            
            # Get all unique tags across all files
            all_tags = set()
            for data in all_data:
                all_tags.update(data.keys())
            
            for tag in all_tags:
                # Get data for this tag from all files
                tag_data_list = []
                for data in all_data:
                    if tag in data:
                        tag_data_list.append(data[tag])
                
                if tag_data_list:
                    print(f"Plotting: {tag}")
                    tag_data = tag_data_list[0]
                    if tag_data['type'] == 'scalar':
                        plot_scalar_data(tag, tag_data_list, folder_names, show_plot=True, smoothing_factor=smoothing_factor)
                    elif tag_data['type'] == 'histogram':
                        plot_histogram_data(tag, tag_data_list, folder_names, show_plot=True, smoothing_factor=smoothing_factor)
                    elif tag_data['type'] == 'image':
                        plot_image_data(tag, tag_data_list, folder_names, show_plot=True, smoothing_factor=smoothing_factor)
    else:
        # Interactive mode
        interactive_plot_interface(all_data, folder_names)

if __name__ == "__main__":
    main() 