#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys
from mnist import prune_pareto, DetectorResult
import pickle


if __name__ == "__main__":
    for dataset_name in [sys.argv[1]]:
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots(figsize=[5, 5], layout="tight")
        fig3, ax3 = plt.subplots()
        fig4, ax4 = plt.subplots()
        fig5, ax5 = plt.subplots()
        fig6, ax6 = plt.subplots()
        all_location_errors = []
        all_detection_delays = []
        all_run_times = []
        latex_table = ""
        xticklabels = []
        for detector_name in ["HDDDM", "HDDDM_N", "MMDDDM", "MMDDDM_N", "MMDDDM_E", "MMDDDM_NE", "D3", "D3_N", "D3_E", "D3_NE", "SpectralDDM"]:
            try:
                with open(detector_name + "_" + dataset_name + "_pareto_frontier.pickle", "rb") as f:
                    pareto_frontier: list[DetectorResult] = pickle.load(file=f)
                prune_pareto(pareto_frontier, lambda item1, item2: ((item1.false_negatives <= item2.false_negatives and item1.false_positives < item2.false_positives) or
                                                                    (item1.false_negatives < item2.false_negatives and item1.false_positives <= item2.false_positives) or
                                                                    (item1.false_negatives <= item2.false_negatives and item1.false_positives <= item2.false_positives and not (item1.location_errors > item2.location_errors))))
                F1_scores = []
                location_errors = []
                location_errors_stds = []
                detection_delays = []
                run_times = []
                false_negatives = []
                false_positives = []
                false_negatives_stds = []
                false_positives_stds = []
                param1 = []
                param2 = []
                fewest_misclassification = pareto_frontier[0]
                for s in pareto_frontier:
                    print(s.detection_name, s.parameters, "false_negatives=", s.false_negatives, "false_positives=", s.false_positives, "location_errors=", s.location_errors, "detection_delays=", s.detection_delays)
                    location_errors.append(s.location_errors)
                    location_errors_stds.append(s.location_errors_std)
                    detection_delays.append(s.detection_delays)
                    F1_scores.append((2.0*s.true_positives)/(2.0*s.true_positives+s.false_positives+s.false_negatives))
                    run_times.append(s.run_time)
                    false_negatives.append(s.false_negatives)
                    false_positives.append(s.false_positives)
                    false_negatives_stds.append(s.false_negatives_std)
                    false_positives_stds.append(s.false_positives_std)
                    if detector_name[:3] == "MMD":
                        param1.append(s.parameters["batching_size"])
                        param2.append(s.parameters["gamma"])
                        detector_color = "tab:green"
                    elif detector_name[:5] == "HDDDM":
                        param1.append(s.parameters["batching_size"])
                        param2.append(s.parameters["gamma"])
                        detector_color = "tab:orange"
                    elif detector_name[:2] == "D3":
                        param1.append(s.parameters["window_size"])
                        param2.append(s.parameters["auc_threshold"])
                        detector_color = "tab:red"
                    elif detector_name[:8] == "Spectral":
                        param1.append(s.parameters["n_eigen"])
                        param2.append(s.parameters["test_size"])
                        detector_color = "tab:blue"
                    if (s.false_positives + s.false_negatives) < (fewest_misclassification.false_positives + fewest_misclassification.false_negatives):
                        fewest_misclassification = s
                if detector_name[-2:] == "_N":  # no localization
                    latex_detector_name = detector_name[:-2]
                    detector_style = "-"
                elif detector_name[-2:] == "_E":  # with localization, stride 1
                    latex_detector_name = detector_name[:-2]+"$\odot$1"
                    detector_style = "-."
                elif detector_name[-3:] == "_NE":  # no localization, stride 1
                    latex_detector_name = detector_name[:-3]+" 1"
                    detector_style = ":"
                elif "_" not in detector_name and not detector_name == "SpectralDDM":
                    latex_detector_name = detector_name+"$\odot$"
                    detector_style = "--"
                elif detector_name == "SpectralDDM":
                    latex_detector_name = "SDDM"
                    detector_style = "-"
                else:
                    print("Bad detector name", detector_name)
                    exit()
                latex_table += (" & "+latex_detector_name+" & "+str(round(fewest_misclassification.location_errors, 1))+" &\(\pm\) "+str(round(fewest_misclassification.location_errors_std, 1))+" & " +
                                str(round(fewest_misclassification.detection_delays, 1))+" &\(\pm\) "+str(round(fewest_misclassification.detection_delays_std, 1))+" & " +
                                str(round(fewest_misclassification.run_time, 1))+" &\(\pm\) "+str(round(fewest_misclassification.run_time_std, 1))+" \\\\ % "+
                                str(fewest_misclassification.parameters)+" fp="+str(fewest_misclassification.false_positives)+" fn="+str(fewest_misclassification.false_negatives)+"\n")
                all_location_errors.append(np.array(location_errors)[~np.isnan(location_errors)])
                all_detection_delays.append(np.array(detection_delays)[~np.isnan(detection_delays)])
                all_run_times.append(run_times)
                ax1.plot(np.array(false_negatives)-np.array(false_positives), location_errors, label=detector_name)
                ax2.plot(false_negatives, false_positives, label=latex_detector_name, color=detector_color, linestyle=detector_style)
                ax3.plot(param2, param1, label=detector_name, color=detector_color, linestyle=detector_style)
                xticklabels.append(detector_name)
            except FileNotFoundError:
                pass
        ax1.set_xlim([-2, 2])
        ax1.set_ylabel("location error")
        ax2.set_xlabel("false negatives")
        ax2.set_ylabel("false positives")
        ax2.set_aspect("equal")
        ax3.legend()
        ax4.boxplot(all_location_errors)
        ax4.set_xticklabels(xticklabels)
        ax4.set_ylabel("location errors")
        ax5.boxplot(all_detection_delays)
        ax5.set_xticklabels(xticklabels)
        ax5.set_ylabel("detection delays")
        ax6.boxplot(all_run_times)
        ax6.set_xticklabels(xticklabels)
        ax6.set_ylabel("run time")
        save_filename = dataset_name + "_pareto_frontier" + ".pdf"
        if dataset_name == "mnist":
            ax2.set_xlim([0, min(0.8, ax2.get_xlim()[1])])
            ax2.set_ylim([0, min(0.8, ax2.get_ylim()[1])])
        elif dataset_name == "rialto":
            ax2.set_xlim([0, min(1.0, ax2.get_xlim()[1])])
            ax2.set_ylim([0, min(1.0, ax2.get_ylim()[1])])
        elif dataset_name == "covtype":
            if False:
                ax2.set_xlim([0, min(1.2, ax2.get_xlim()[1])])
                ax2.set_ylim([0, min(1.2, ax2.get_ylim()[1])])
                save_filename = "covtype_pareto_frontier_large.pdf"
            else:
                ax2.set_xlim([0, min(0.3, ax2.get_xlim()[1])])
                ax2.set_ylim([0, min(0.3, ax2.get_ylim()[1])])
                save_filename = "covtype_pareto_frontier_close.pdf"
        elif dataset_name == "music":
            ax2.set_xlim([0, min(2.0, ax2.get_xlim()[1])])
            ax2.set_ylim([0, min(2.0, ax2.get_ylim()[1])])
        ax1.legend()
        ax2.legend()
        print(latex_table)
        fig2.savefig(save_filename)
        plt.show()
    exit()
