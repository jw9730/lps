# pylint: disable=line-too-long
import sys
from pathlib import Path
from matplotlib import pyplot as plt, colors as mcolors, cm
import numpy as np


def main():
    result_dir = Path('experiments/exp_classify_analysis_results')
    result_dir.mkdir(exist_ok=True, parents=True)

    inference_sample_size_list = [1, 2, 5, 10, 20, 50, 100, 200]
    output_variance_ga = [
        3.0706462169081845e-07,
        1.5521085427016174e-07,
        6.171463413547161e-08,
        3.017349290337683e-08,
        1.5502213285103925e-08,
        6.172048835150004e-09,
        3.0883258973997898e-09,
        1.5264307195648647e-09
    ]
    output_variance_ps = [
        1.6541861126420024e-07,
        8.08703081663165e-08,
        3.34797365137036e-08,
        1.626220088479192e-08,
        8.201903479779489e-09,
        3.3071799775451203e-09,
        1.6516958891311414e-09,
        8.255185086715526e-10
    ]
    loss_variance_ga = [
        1.2749933750875217e-06,
        6.416980301819289e-07,
        2.5513579288871584e-07,
        1.2498019269397055e-07,
        6.41882245852448e-08,
        2.5605660191740127e-08,
        1.2772980042971844e-08,
        6.332357995155924e-09
    ]
    loss_variance_ps = [
        6.86501097257535e-07,
        3.348565073013885e-07,
        1.3830502426699794e-07,
        6.736574970622854e-08,
        3.401901327210593e-08,
        1.366817101688436e-08,
        6.829182854227606e-09,
        3.416330793589359e-09
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    (ax1, ax2) = axes
    # output variance
    ax1.plot(inference_sample_size_list, output_variance_ga, label='GA', color='C0')
    ax1.plot(inference_sample_size_list, output_variance_ps, label='PS (Ours)', color='C1')
    ax1.legend()
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(1, 200)
    ax1.set_xticks([1, 10, 100, 200], [1, 10, 100, 200])
    ax1.set_xlabel('Inference sample size')
    ax1.set_title('Output variance', fontsize='medium')
    # loss variance
    ax2.plot(inference_sample_size_list, loss_variance_ga, label='GA', color='C0')
    ax2.plot(inference_sample_size_list, loss_variance_ps, label='PS (Ours)', color='C1')
    ax2.legend()
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(1, 200)
    ax2.set_xticks([1, 10, 100, 200], [1, 10, 100, 200])
    ax2.set_xlabel('Inference sample size')
    ax2.set_title('Loss variance', fontsize='medium')
    plt.savefig(result_dir / 'ps_ga_variance.pdf', bbox_inches='tight')

    train_epoch_list = [100, 500, 1000, 1500, 2000]
    grad_direction_norm_ga = [
        0.014915736392140388,
        4.267692565917969e-05,
        0.00017434358596801758,
        0.000244140625,
        0.0001653432846069336,
    ]
    grad_direction_norm_ps = [
        1.0259549617767334,
        0.2769050896167755,
        0.7440996766090393,
        9.538859367370605,
        5.858701229095459,
    ]
    # gradient direction norm
    fig, ax = plt.subplots(figsize=(3, 3))
    plt.plot(train_epoch_list, grad_direction_norm_ga, label='GA', color='C0')
    plt.plot(train_epoch_list, grad_direction_norm_ps, label='PS (Ours)', color='C1')
    plt.legend()
    plt.yscale('log')
    plt.xlim(100, 2000)
    plt.xlabel('Train epoch')
    ax.set_xticks([100, 1000, 2000], [100, 1000, 2000])
    plt.title('Gradient direction norm', fontsize='medium')
    plt.savefig(result_dir / 'ps_ga_grad_norm.pdf', bbox_inches='tight')

    train_sample_size_list = [1, 2, 5, 10, 20, 50]
    inference_sample_size_list = [1, 2, 5, 10, 20, 50, 100, 200]
    test_accuracy = [97.5, 99.5, 100, 100, 99, 94]
    aggregated_permutation_entropy = [
        4.974285411834717,
        5.299808316230774,
        5.358375126123429,
        5.742276425361633,
        5.747604515552521,
        5.30451651096344
    ]
    output_variance = [
        {1: 0.05728540128145767, 2: 0.032672350350251754, 5: 0.012965064540418481, 10: 0.006170224073062822, 20: 0.002819192289717801, 50: 0.0010073832749461717, 100: 0.0004884824521033333, 200: 0.00024102387989053761},
        {1: 0.04755314971920915, 2: 0.016754518029810678, 5: 0.001946336805543455, 10: 0.0002948745111992412, 20: 4.783197569895825e-05, 50: 6.661319064728145e-06, 100: 1.78873204204653e-06, 200: 6.23118489596231e-07},
        {1: 0.09002647590863345, 2: 0.03659204683170373, 5: 0.0036435529535214055, 10: 0.00023107155553143865, 20: 1.0280959465683454e-05, 50: 1.0580583547434355e-06, 100: 5.617410243095228e-08, 200: 4.527922321175236e-08},
        {1: 0.14569390694058423, 2: 0.08929549883315309, 5: 0.02144938430969646, 10: 0.002310744434182581, 20: 5.360684752834944e-05, 50: 5.807636329062895e-06, 100: 5.091100683234808e-06, 200: 4.461098718287459e-06},
        {1: 0.1241547902609871, 2: 0.09962326593091256, 5: 0.04785178475418612, 10: 0.014283032962366656, 20: 0.0020969656284951866, 50: 3.715859175285424e-05, 100: 4.112851225488069e-06, 200: 1.6563160001433478e-06},
        {1: 0.08050555874827163, 2: 0.09401509907651842, 5: 0.08431665470390054, 10: 0.04340998786275198, 20: 0.013189855166497915, 50: 0.00038904690386628346, 100: 2.4882944106092414e-05, 200: 4.056367785957766e-06}
    ]
    loss_variance = [
        {1: 0.4000418267228548, 2: 0.13337507248833302, 5: 0.03486299537701681, 10: 0.013175297208119142, 20: 0.005040805204194193, 50: 0.0016127704062788007, 100: 0.000744324748079055, 200: 0.00036107173560480995},
        {1: 0.7236731346015357, 2: 0.1295406887971354, 5: 0.0065013059269430976, 10: 0.0006241246756301404, 20: 0.00010132196784034605, 50: 1.2004449122738783e-05, 100: 1.93354597911522e-06, 200: 6.446190115315819e-07},
        {1: 7.132708199979103, 2: 1.216346814788016, 5: 0.04653701569269349, 10: 0.005274886205105127, 20: 0.0014057028670501702, 50: 0.0004878179465428622, 100: 0.0002880938077497577, 200: 0.00016064789667179618},
        {1: 52.28317690177816, 2: 13.798393484290461, 5: 1.1953691224350025, 10: 0.09047484326610364, 20: 0.008144452601046978, 50: 0.0017325448479229546, 100: 0.0004526347949273957, 200: 0.0004094342561653763},
        {1: 41.50281031121177, 2: 15.521975466521464, 5: 2.9619437042459076, 10: 0.5271301576410243, 20: 0.047739374706024365, 50: 0.004049775145598979, 100: 0.002103717963976271, 200: 0.0010971306150064344},
        {1: 160.86562676302051, 2: 75.47981941667413, 5: 21.206526386592667, 10: 5.923607240395657, 20: 1.0606971534114171, 50: 0.048946166348650025, 100: 0.013917550760540381, 200: 0.008890745940731463}
    ]
    loss_mean = [
        {1: 0.435622586235404, 2: 0.29533577896188945, 5: 0.19525568856319298, 10: 0.16018273997702637, 20: 0.14125716980728611, 50: 0.12965848148473924, 100: 0.12588263908468433, 200: 0.12427737602708021},
        {1: 0.3065085884765449, 2: 0.11678855236912, 5: 0.02751179982636671, 10: 0.01074560106758589, 20: 0.0058102655517033594, 50: 0.003877121023839556, 100: 0.0032957341034962197, 200: 0.0030892978498623237},
        {1: 0.8668395363446325, 2: 0.3046738412324518, 5: 0.08010372936664639, 10: 0.05043500425110715, 20: 0.04305245564993055, 50: 0.03773379633310266, 100: 0.04095681383465477, 200: 0.03911080474429121},
        {1: 2.897936876192689, 2: 1.26465321514057, 5: 0.2714503089405946, 10: 0.09756674202105552, 20: 0.05137166503250737, 50: 0.02990997344752923, 100: 0.016784781603436136, 200: 0.01457264707265697},
        {1: 3.0712811067467554, 2: 1.7953001398764536, 5: 0.6234068413883885, 10: 0.23779508759592047, 20: 0.09143870905411093, 50: 0.05856174626692872, 100: 0.059284366246608246, 200: 0.05450170620723685},
        {1: 4.786832296289504, 2: 3.640420485427603, 5: 1.9148857730797317, 10: 0.900712268892459, 20: 0.3894244514891711, 50: 0.19439654105668588, 100: 0.1990255922348682, 200: 0.19358754464953973},
    ]
    # test accuracy
    fig, ax = plt.subplots(figsize=(3, 3))
    plt.axhline(y=100, color='k', linestyle='--', linewidth=0.5)
    plt.plot(train_sample_size_list, test_accuracy, '-o', label='Test accuracy', color='C1')
    plt.xscale('log')
    plt.xlabel('Train sample size')
    plt.title('Test accuracy (%)', fontsize='medium')
    ax.set_yticks([90, 95, 100], [90, 95, 100])
    ax.set_xticks([1, 10, 50], [1, 10, 50])
    plt.ylim(89, 101)
    plt.xlim(1, 50)
    plt.savefig(result_dir / 'train_sample_size_accuracy.pdf', bbox_inches='tight')
    # aggregated permutation entropy
    fig, ax = plt.subplots(figsize=(3, 3))
    # dashed horizontal line at random entropy
    plt.axhline(y=7.035248444080353, linestyle='--', color='C0', label='Random')
    plt.plot(train_sample_size_list, aggregated_permutation_entropy, '-o', color='C1')
    plt.xscale('log')
    plt.xlabel('Train sample size')
    plt.title('Entropy of sampled permutation', fontsize='medium')
    ax.set_xticks([1, 10, 50], [1, 10, 50])
    plt.xlim(1, 50)
    plt.savefig(result_dir / 'train_sample_size_entropy.pdf', bbox_inches='tight')
    # output variance, loss variance, loss mean
    normalize = mcolors.Normalize(vmin=np.log10(1), vmax=np.log10(200))
    colormap = cm.viridis
    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalarmappaple.set_array([np.log10(i) for i in inference_sample_size_list])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3))
    (ax1, ax2, ax3) = axes
    # output variance
    ax1.plot(train_sample_size_list, [e[1] for e in output_variance], '-o', label='1', color=colormap(normalize(np.log10(1))), linewidth=0.8, markersize=4)
    ax1.plot(train_sample_size_list, [e[2] for e in output_variance], '-o', label='2', color=colormap(normalize(np.log10(2))), linewidth=0.8, markersize=4)
    ax1.plot(train_sample_size_list, [e[5] for e in output_variance], '-o', label='5', color=colormap(normalize(np.log10(5))), linewidth=0.8, markersize=4)
    ax1.plot(train_sample_size_list, [e[10] for e in output_variance], '-o', label='10', color=colormap(normalize(np.log10(10))), linewidth=0.8, markersize=4)
    ax1.plot(train_sample_size_list, [e[20] for e in output_variance], '-o', label='20', color=colormap(normalize(np.log10(20))), linewidth=0.8, markersize=4)
    ax1.plot(train_sample_size_list, [e[50] for e in output_variance], '-o', label='50', color=colormap(normalize(np.log10(50))), linewidth=0.8, markersize=4)
    ax1.plot(train_sample_size_list, [e[100] for e in output_variance], '-o', label='100', color=colormap(normalize(np.log10(100))), linewidth=0.8, markersize=4)
    ax1.plot(train_sample_size_list, [e[200] for e in output_variance], '-o', label='200', color=colormap(normalize(np.log10(200))), linewidth=0.8, markersize=4)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Train sample size')
    ax1.set_title('Output variance', fontsize='medium')
    ax1.set_xticks([1, 10, 50], [1, 10, 50])
    # loss variance
    ax2.plot(train_sample_size_list, [e[1] for e in loss_variance], '-o', label='1', color=colormap(normalize(np.log10(1))), linewidth=0.8, markersize=4)
    ax2.plot(train_sample_size_list, [e[2] for e in loss_variance], '-o', label='2', color=colormap(normalize(np.log10(2))), linewidth=0.8, markersize=4)
    ax2.plot(train_sample_size_list, [e[5] for e in loss_variance], '-o', label='5', color=colormap(normalize(np.log10(5))), linewidth=0.8, markersize=4)
    ax2.plot(train_sample_size_list, [e[10] for e in loss_variance], '-o', label='10', color=colormap(normalize(np.log10(10))), linewidth=0.8, markersize=4)
    ax2.plot(train_sample_size_list, [e[20] for e in loss_variance], '-o', label='20', color=colormap(normalize(np.log10(20))), linewidth=0.8, markersize=4)
    ax2.plot(train_sample_size_list, [e[50] for e in loss_variance], '-o', label='50', color=colormap(normalize(np.log10(50))), linewidth=0.8, markersize=4)
    ax2.plot(train_sample_size_list, [e[100] for e in loss_variance], '-o', label='100', color=colormap(normalize(np.log10(100))), linewidth=0.8, markersize=4)
    ax2.plot(train_sample_size_list, [e[200] for e in loss_variance], '-o', label='200', color=colormap(normalize(np.log10(200))), linewidth=0.8, markersize=4)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Train sample size')
    ax2.set_title('Loss variance', fontsize='medium')
    ax2.set_xticks([1, 10, 50], [1, 10, 50])
    # loss mean
    ax3.plot(train_sample_size_list, [e[1] for e in loss_mean], '-o', label='1', color=colormap(normalize(np.log10(1))), linewidth=0.8, markersize=4)
    ax3.plot(train_sample_size_list, [e[2] for e in loss_mean], '-o', label='2', color=colormap(normalize(np.log10(2))), linewidth=0.8, markersize=4)
    ax3.plot(train_sample_size_list, [e[5] for e in loss_mean], '-o', label='5', color=colormap(normalize(np.log10(5))), linewidth=0.8, markersize=4)
    ax3.plot(train_sample_size_list, [e[10] for e in loss_mean], '-o', label='10', color=colormap(normalize(np.log10(10))), linewidth=0.8, markersize=4)
    ax3.plot(train_sample_size_list, [e[20] for e in loss_mean], '-o', label='20', color=colormap(normalize(np.log10(20))), linewidth=0.8, markersize=4)
    ax3.plot(train_sample_size_list, [e[50] for e in loss_mean], '-o', label='50', color=colormap(normalize(np.log10(50))), linewidth=0.8, markersize=4)
    ax3.plot(train_sample_size_list, [e[100] for e in loss_mean], '-o', label='100', color=colormap(normalize(np.log10(100))), linewidth=0.8, markersize=4)
    ax3.plot(train_sample_size_list, [e[200] for e in loss_mean], '-o', label='200', color=colormap(normalize(np.log10(200))), linewidth=0.8, markersize=4)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('Train sample size')
    ax3.set_title('Loss mean', fontsize='medium')
    ax3.set_xticks([1, 10, 50], [1, 10, 50])
    # global colorbar
    cbar = fig.colorbar(scalarmappaple, ax=axes.ravel().tolist(), label='Inference sample size')
    cbar.set_ticks([np.log10(i) for i in [1, 10, 50, 200]])
    cbar.set_ticklabels([1, 10, 50, 200])
    plt.savefig(result_dir / 'train_sample_size_variance.pdf', bbox_inches='tight')

    # done
    sys.exit()


if __name__ == '__main__':
    main()
