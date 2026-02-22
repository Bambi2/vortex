import sys
import time
import mne
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from csp import CSP

mne.set_log_level('ERROR')

DATASET_DIR="dataset"
RUNS_BY_EXPERIMENT = [[3, 7], [4, 8], [5, 9], [6, 10], [3, 7, 11]]
# RUNS_BY_EXPERIMENT = [[3], [4], [5], [6], [7]]
NUMBER_OF_SUBJECTS = 109

def load_raw(subject: int, run: int, preload: bool = True):
    filepath = f"{DATASET_DIR}/S{subject:03d}/S{subject:03d}R{run:02d}.edf"

    try:
        raw = mne.io.read_raw_edf(filepath, preload=preload)
    except FileNotFoundError:
        print("File not found")
        sys.exit(1)

    # Standardization to rename channels and correct positioning
    mne.datasets.eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="ignore")

    return raw

def visualize(raw: mne.io.BaseRaw):
    print(raw.info)
    raw.plot()
    raw.compute_psd().plot()

    raw.filter(8.0, 30.0)

    raw.compute_psd().plot(average=True)

    plt.show()

def train(raws: list, verbose: bool = True):
    all_x = []
    all_y = []
    for raw in raws:
        raw.filter(8.0, 30.0)
        epochs = mne.Epochs(raw, event_id=['T1', 'T2'], tmin=0.0, tmax=4.0, baseline=None)
        all_x.append(epochs.get_data())
        all_y.append(epochs.events[:, -1])

    x = np.concatenate(all_x)
    y = np.concatenate(all_y)

    pipeline = sklearn.pipeline.Pipeline([
        ('CSP', CSP(n_components=4)),
        ('LDA', sklearn.discriminant_analysis.LinearDiscriminantAnalysis())
    ])

    n_splits = min(5, len(x) // 2)
    if n_splits <= 2:
        if verbose: print("The dataset is too small")
        return None

    scores = sklearn.model_selection.cross_val_score(pipeline, x, y, cv=n_splits)
    if verbose:
        print(scores)
        print(f"cross_val_score: {scores.mean()}")

    return scores.mean()

def predict(raw: mne.io.BaseRaw, subject: int, run: int):
    raw.filter(8.0, 30.0)
    epochs = mne.Epochs(raw, event_id=['T1', 'T2'], tmin=0.0, tmax=4.0, baseline=None)

    split_index = int(len(epochs.events) * 0.7)
    x_train = epochs.get_data()[:split_index]
    y_train = epochs.events[:split_index, -1]
    test_events = epochs.events[split_index:]

    pipeline = sklearn.pipeline.Pipeline([
        ('CSP', CSP(n_components=4)),
        ('LDA', sklearn.discriminant_analysis.LinearDiscriminantAnalysis())
    ])
    pipeline.fit(x_train, y_train)

    raw_stream = load_raw(subject, run, preload=False)
    sfreq = raw_stream.info['sfreq']
    number_of_samples_per_event = int(4.0 * sfreq)
    correct_count = 0

    print(f"{'epoch':>5}  {'prediction':>10}  {'truth':>5}  {'equal':>5}  {'time':>6}")
    print(f"{'-----':>5}  {'----------':>10}  {'-----':>5}  {'-----':>5}  {'------':>6}")
    for i, event in enumerate(test_events):
        start = time.time()

        data = raw_stream.get_data(start=event[0], stop=event[0] + number_of_samples_per_event)
        data = mne.filter.filter_data(data, sfreq, l_freq=8.0, h_freq=30.0)

        prediction = pipeline.predict(data[np.newaxis, :])[0]

        elapsed = time.time() - start

        is_correct = prediction == event[2]
        if is_correct:
            correct_count += 1

        print(f"{i:>5}  {prediction:>10}  {event[2]:>5}  {str(is_correct):>5}  {elapsed:>5.3f}s")

    accuracy = correct_count / len(test_events) if len(test_events) > 0 else 0
    print(f"\nAccuracy: {accuracy:.4f}")

def run_all_subjects_experiments():
    means = []
    for experiment, runs in enumerate(RUNS_BY_EXPERIMENT):
        accuracies = []
        for subject in range(1, NUMBER_OF_SUBJECTS+1):
            raws = [load_raw(subject, run) for run in runs]
            accuracy = train(raws, verbose=False)
            if accuracy is None:
                continue
            accuracies.append(accuracy)
            print(f"experiment {experiment}: subject {subject}: accuracy = {accuracy:0.4f}")
        
        means.append(np.mean(accuracies))
    
    print(f"\nMean accuracy of the {len(RUNS_BY_EXPERIMENT)} different experiments for all {NUMBER_OF_SUBJECTS} subjects:")

    for experiment in range(len(RUNS_BY_EXPERIMENT)):
        print(f"experiment {experiment} (runs {RUNS_BY_EXPERIMENT[experiment]}): accuracy = {means[experiment]:0.4f}")
    
    print(f"\nMean accuracy of {len(RUNS_BY_EXPERIMENT)} experiments: {np.mean(means):0.4f}")


def main():
    if len(sys.argv) == 1:
        run_all_subjects_experiments()
        return
    
    if len(sys.argv) != 4:
        print("Wrong number of arguments.")
        print("Example of usage: python mybci.py 1 3 visualize")
        sys.exit(1)

    subject = int(sys.argv[1])
    run = int(sys.argv[2])
    mode = sys.argv[3]

    if run in (1, 2) and mode in ("predict", "train"):
        print("The runs are not for predicring and training as they contain only resting.")
        sys.exit(1)

    raw = load_raw(subject, run)

    if mode == "visualize":
        visualize(raw)
    elif mode == "train":
        train([raw])
    elif mode == "predict":
        predict(raw, subject, run)
    else:
        print(f"Unknown mode: {mode}. Use 'visualize', 'train', 'predict' or without arguments.")
        sys.exit(1)

if __name__ == "__main__":
    main()