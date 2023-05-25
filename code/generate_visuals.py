import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
mpl.font_manager._rebuild()
import matplotlib
# Use the newly integrated Roboto font family for all text.
plt.rc('font', family='Open Sans')

fig, ax = plt.subplots()
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(7, 4.5)

labels = ["DP", "WDP", "AdaGrad", "PGD", "WMD"]
x = np.arange(len(labels))
width = .3
colors = ["#648FFF", "#785EF0", "#DC267F", '#FE6100', "#FFB000"]

colors = ["#648FFF", "#785EF0", "#DC267F", '#FE6100', "#FFB000"]
# Save the chart so we can loop through the bars below.
robustaqbars = ax.bar(
		x=x - width,
		height=[14.2, 14.3, 0.81, 13.9, 13.5],
		label="Robust",
		width = .3
)
lassoaqbars = ax.bar(
		x=x,
		height=[12.1, 12.1, .73, 10.1, 8.58],
	width=.3,
	label="Lasso"
)

asymetricaqbars = ax.bar(
		x=x+width,
		height=[14.1, 14.1, .65, 12.9, 12.6],
	width=.3,
	label="Lasso"
)
ax.set_xticks(x, labels)
# Axis formatting.
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.tick_params(bottom=False, left=False)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.grid(False)

#Times: ada: .6, Prox: 8.60
#Gaps: ada: -.73, prox: -10.00, mirror: -8.58, dp: -12, wdp:-12
# Add text annotations to the top of the bars.

for idx, bar in enumerate(lassoaqbars):
	bar_color = lassoaqbars[idx].get_facecolor()
	ax.text(
			bar.get_x() + bar.get_width() / 2,
			bar.get_height() + 0.3,
			round(bar.get_height(), 1),
			horizontalalignment='center',
			color=bar_color,
			weight='bold'
	)

	for idx, bar in enumerate(asymetricaqbars):
		bar_color = asymetricaqbars[idx].get_facecolor()
		ax.text(
				bar.get_x() + bar.get_width() / 2,
				bar.get_height() + 0.3,
				round(bar.get_height(), 1),
				horizontalalignment='center',
				color=bar_color,
				weight='bold'
	)

ax.set_xticklabels(labels)

for idx, bar in enumerate(robustaqbars):
	bar_color = robustaqbars[idx].get_facecolor()
	ax.text(
			bar.get_x() + bar.get_width() / 2,
			bar.get_height() + 0.3,
			round(bar.get_height(), 1),
			horizontalalignment='center',
			color=bar_color,
			weight='bold'
	)


# Add labels and a title.
ax.set_xlabel('Numerical Continuation Algorithm', labelpad=15, color='#333333')
ax.set_ylabel('Inverse Log of Approximation Error', labelpad=15, color='#333333')
ax.set_title('Approximation Error of \nSeveral Numerical Continuation Algorithms for Robust Pass', pad=15, color='#333333',
						 weight='bold')

fig.tight_layout()

plt.savefig("images/aq_robust.png")