import numpy as np
import matplotlib.pyplot as plt
import scipy
from   scipy.integrate import simps

"""| Number | Velocity | First spectrum | Second spectrum | Third spectrum | Fourth spectrum |"""

data = []
with open('n4388_3.asc') as fobj:
    for line in fobj:
        row = line.split()
        data.append(row)

spectrum_saved_without_noise_1 = [0 for i in range(256)]
spectrum_saved_without_noise_2 = [0 for i in range(256)]
spectrum_saved_without_noise_3 = [0 for i in range(256)]
spectrum_saved_without_noise_4 = [0 for i in range(256)]

spectrum_saved_with_noise_1 = [0 for i in range(256)]
spectrum_saved_with_noise_2 = [0 for i in range(256)]
spectrum_saved_with_noise_3 = [0 for i in range(256)]
spectrum_saved_with_noise_4 = [0 for i in range(256)]

for w in range(18):

    velocity              = []
    spectrum_with_noise_1 = []
    spectrum_with_noise_2 = []
    spectrum_with_noise_3 = []
    spectrum_with_noise_4 = []
    for i in range(0,256):

        velocity.append(float(data[i+1][1]))
        spectrum_with_noise_1.append(float(data[i+257*w+1][2]))
        spectrum_with_noise_2.append(float(data[i+257*w+1][3]))
        spectrum_with_noise_3.append(float(data[i+257*w+1][4]))
        spectrum_with_noise_4.append(float(data[i+257*w+1][5]))

        spectrum_saved_with_noise_1[i] =spectrum_saved_with_noise_1[i]+ float(data[i+1][2])/18
        spectrum_saved_with_noise_2[i] =spectrum_saved_with_noise_2[i]+ float(data[i+1][3])/18
        spectrum_saved_with_noise_3[i] =spectrum_saved_with_noise_3[i]+ float(data[i+1][4])/18
        spectrum_saved_with_noise_4[i] =spectrum_saved_with_noise_4[i]+ float(data[i+1][5])/18


    mean_value_spectrum_with_noise_1 = np.mean(spectrum_with_noise_1)
    mean_value_spectrum_with_noise_2 = np.mean(spectrum_with_noise_2)
    mean_value_spectrum_with_noise_3 = np.mean(spectrum_with_noise_3)
    mean_value_spectrum_with_noise_4 = np.mean(spectrum_with_noise_4)

    standard_deviation_spectrum_with_noise_1 = np.std(spectrum_with_noise_1)
    standard_deviation_spectrum_with_noise_2 = np.std(spectrum_with_noise_2)
    standard_deviation_spectrum_with_noise_3 = np.std(spectrum_with_noise_3)
    standard_deviation_spectrum_with_noise_4 = np.std(spectrum_with_noise_4)

    spectrum_without_noise_1 = []
    spectrum_without_noise_2 = []
    spectrum_without_noise_3 = []
    spectrum_without_noise_4 = []

    for i in range(256):
        if abs(spectrum_with_noise_1[i] - standard_deviation_spectrum_with_noise_1)>= 2*standard_deviation_spectrum_with_noise_1:
            spectrum_without_noise_1.append(mean_value_spectrum_with_noise_1)
        else:
            spectrum_without_noise_1.append(spectrum_with_noise_1[i])

        if abs(spectrum_with_noise_2[i] - standard_deviation_spectrum_with_noise_2)>= 2*standard_deviation_spectrum_with_noise_2:
            spectrum_without_noise_2.append(mean_value_spectrum_with_noise_2)
        else:
            spectrum_without_noise_2.append(spectrum_with_noise_2[i])

        if abs(spectrum_with_noise_3[i] - standard_deviation_spectrum_with_noise_3)>= 2*standard_deviation_spectrum_with_noise_3:
            spectrum_without_noise_3.append(mean_value_spectrum_with_noise_3)
        else:
            spectrum_without_noise_3.append(spectrum_with_noise_3[i])

        if abs(spectrum_with_noise_4[i] - standard_deviation_spectrum_with_noise_4)>= 2*standard_deviation_spectrum_with_noise_4:
            spectrum_without_noise_4.append(mean_value_spectrum_with_noise_4)
        else:
            spectrum_without_noise_4.append(spectrum_with_noise_4[i])

    interval_HI_start = 80
    interval_HI_end   = 180
    size_interval_HI  = abs(interval_HI_start-interval_HI_end)

    spectrum_with_linear_regression_1 = []
    spectrum_with_linear_regression_2 = []
    spectrum_with_linear_regression_3 = []
    spectrum_with_linear_regression_4 = []

    mean_value_linear_regression_left_1  = 0
    mean_value_linear_regression_right_1 = 0
    mean_value_linear_regression_left_2  = 0
    mean_value_linear_regression_right_2 = 0
    mean_value_linear_regression_left_3  = 0
    mean_value_linear_regression_right_3 = 0
    mean_value_linear_regression_left_4  = 0
    mean_value_linear_regression_right_4 = 0

    for i in range(interval_HI_start,interval_HI_start+5):
        mean_value_linear_regression_left_1 = mean_value_linear_regression_left_1+spectrum_without_noise_1[i]/5
        mean_value_linear_regression_left_2 = mean_value_linear_regression_left_2+spectrum_without_noise_2[i]/5
        mean_value_linear_regression_left_3 = mean_value_linear_regression_left_3+spectrum_without_noise_3[i]/5
        mean_value_linear_regression_left_4 = mean_value_linear_regression_left_4+spectrum_without_noise_4[i]/5

    for i in range(interval_HI_end-5,interval_HI_end):
        mean_value_linear_regression_right_1=mean_value_linear_regression_right_1+spectrum_without_noise_1[i]/5
        mean_value_linear_regression_right_2=mean_value_linear_regression_right_2+spectrum_without_noise_2[i]/5
        mean_value_linear_regression_right_3=mean_value_linear_regression_right_3+spectrum_without_noise_3[i]/5
        mean_value_linear_regression_right_4=mean_value_linear_regression_right_4+spectrum_without_noise_4[i]/5

    for i in range(256):
        if (i>=interval_HI_start) and (i<=interval_HI_end):
            spectrum_with_linear_regression_1.append( ((mean_value_linear_regression_right_1-mean_value_linear_regression_left_1)/(velocity[interval_HI_end]-velocity[interval_HI_start]))*velocity[i]+mean_value_linear_regression_left_1-(velocity[interval_HI_start])*((mean_value_linear_regression_right_1-mean_value_linear_regression_left_1)/(velocity[interval_HI_end]-velocity[interval_HI_start]))  )
            spectrum_with_linear_regression_2.append( ((mean_value_linear_regression_right_2-mean_value_linear_regression_left_2)/(velocity[interval_HI_end]-velocity[interval_HI_start]))*velocity[i]+mean_value_linear_regression_left_2-(velocity[interval_HI_start])*((mean_value_linear_regression_right_2-mean_value_linear_regression_left_2)/(velocity[interval_HI_end]-velocity[interval_HI_start]))  )
            spectrum_with_linear_regression_3.append( ((mean_value_linear_regression_right_3-mean_value_linear_regression_left_3)/(velocity[interval_HI_end]-velocity[interval_HI_start]))*velocity[i]+mean_value_linear_regression_left_3-(velocity[interval_HI_start])*((mean_value_linear_regression_right_3-mean_value_linear_regression_left_3)/(velocity[interval_HI_end]-velocity[interval_HI_start]))  )
            spectrum_with_linear_regression_4.append( ((mean_value_linear_regression_right_4-mean_value_linear_regression_left_4)/(velocity[interval_HI_end]-velocity[interval_HI_start]))*velocity[i]+mean_value_linear_regression_left_4-(velocity[interval_HI_start])*((mean_value_linear_regression_right_4-mean_value_linear_regression_left_4)/(velocity[interval_HI_end]-velocity[interval_HI_start]))  )
        else:
            spectrum_with_linear_regression_1.append(spectrum_without_noise_1[i])
            spectrum_with_linear_regression_2.append(spectrum_without_noise_2[i])
            spectrum_with_linear_regression_3.append(spectrum_without_noise_3[i])
            spectrum_with_linear_regression_4.append(spectrum_without_noise_4[i])

    polynomial_spectrum_1 = np.polyfit(velocity,spectrum_with_linear_regression_1,3)
    polynomial_spectrum_2 = np.polyfit(velocity,spectrum_with_linear_regression_2,deg=3)
    polynomial_spectrum_3 = np.polyfit(velocity,spectrum_with_linear_regression_3,deg=3)
    polynomial_spectrum_4 = np.polyfit(velocity,spectrum_with_linear_regression_4,deg=3)

    polynomial_valued_spectrum_1 = []
    polynomial_valued_spectrum_2 = []
    polynomial_valued_spectrum_3 = []
    polynomial_valued_spectrum_4 = []

    for i in range(256):
        polynomial_valued_spectrum_1.append(polynomial_spectrum_1[3]+polynomial_spectrum_1[2]*(velocity[i])+polynomial_spectrum_1[1]*(velocity[i]**2)+polynomial_spectrum_1[0]*(velocity[i]**3))
        polynomial_valued_spectrum_2.append(polynomial_spectrum_2[3]+polynomial_spectrum_2[2]*(velocity[i])+polynomial_spectrum_2[1]*(velocity[i]**2)+polynomial_spectrum_2[0]*(velocity[i]**3))
        polynomial_valued_spectrum_3.append(polynomial_spectrum_3[3]+polynomial_spectrum_3[2]*(velocity[i])+polynomial_spectrum_3[1]*(velocity[i]**2)+polynomial_spectrum_3[0]*(velocity[i]**3))
        polynomial_valued_spectrum_4.append(polynomial_spectrum_4[3]+polynomial_spectrum_4[2]*(velocity[i])+polynomial_spectrum_4[1]*(velocity[i]**2)+polynomial_spectrum_4[0]*(velocity[i]**3))


    spectrum_minus_polynomial_1 = []
    spectrum_minus_polynomial_2 = []
    spectrum_minus_polynomial_3 = []
    spectrum_minus_polynomial_4 = []
    for i in range(256):
        spectrum_minus_polynomial_1.append(spectrum_without_noise_1[i]-polynomial_valued_spectrum_1[i])
        spectrum_minus_polynomial_2.append(spectrum_without_noise_2[i]-polynomial_valued_spectrum_2[i])
        spectrum_minus_polynomial_3.append(spectrum_without_noise_3[i]-polynomial_valued_spectrum_3[i])
        spectrum_minus_polynomial_4.append(spectrum_without_noise_4[i]-polynomial_valued_spectrum_4[i])

        spectrum_saved_without_noise_1[i] =spectrum_saved_without_noise_1[i]+ spectrum_minus_polynomial_1[i]/18
        spectrum_saved_without_noise_2[i] =spectrum_saved_without_noise_2[i]+ spectrum_minus_polynomial_2[i]/18
        spectrum_saved_without_noise_3[i] =spectrum_saved_without_noise_3[i]+ spectrum_minus_polynomial_3[i]/18
        spectrum_saved_without_noise_4[i] =spectrum_saved_without_noise_4[i]+ spectrum_minus_polynomial_4[i]/18

true_mean_without_noise = []
true_mean_with_noise    = []
for i in range(256):

    true_mean_without_noise.append((spectrum_saved_without_noise_1[i]+spectrum_saved_without_noise_2[i]+spectrum_saved_without_noise_3[i]+spectrum_saved_without_noise_4[i])/4)
    true_mean_with_noise.append((spectrum_saved_with_noise_1[i]+spectrum_saved_with_noise_2[i]+spectrum_saved_with_noise_3[i]+spectrum_saved_with_noise_4[i])/4)


std_without_noise = np.std(true_mean_without_noise[40:80])
std_with_noise = np.std(true_mean_with_noise[40:80])

"""The quotient signal/noise"""
print('The quotient signal/noise is : ', end='')
print(std_with_noise/std_without_noise)

value_integral_without_noise = simps(true_mean_without_noise,velocity,dx=5.2)
value_integral_with_noise    = simps(true_mean_with_noise,velocity,dx=5.2)

print('Area under the cleaned signal is : ',  end='')
print(abs(value_integral_without_noise)*2)


fig, axs = plt.subplots(2)
fig.suptitle('Signal before and after noise treatment')
axs[0].plot(velocity,true_mean_with_noise,color='red', linewidth=1, label = 'Without treatment')
axs[0].set_ylabel('Intensity')
axs[0].legend(loc="upper right", prop={'size': 9})
axs[1].set_xlabel('Velocity $km/s$')
axs[1].set_ylabel('Intensity')
axs[1].plot(velocity,true_mean_without_noise,color='blue',linewidth=1, label = "With treatment")
axs[1].legend(loc="upper right", prop={'size': 9})
plt.savefig('final_figure.png', dpi=600)
