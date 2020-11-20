plt.rcParams['figure.dpi'] = 120
plt.figure(3)
plt.scatter(x=X_pos[:,0], y=X_pos[:,1], s=10, color='red', marker='*',label='Positive example') # plot the data points from X_pos
plt.scatter(x=X_neg[:,0], y=X_neg[:,1], s=10, color='blue', marker='*',label='Negative exmaple') # plot the data points from X_neg
plt.axis([-15,42,-35,45])
plt.ylabel('Feature 2')
plt.xlabel('Feature 1')
plt.legend(loc='upper right')
c = 2*sigma_neg_sq*sigma_pos_sq*(math.log10(sigma_neg_sq/sigma_pos_sq)) 
x = np.linspace(-30., 30.)
y = np.linspace(-30., 30.)[:, None]

# Decision boundary for unequal diagonal covariance matrix
plt.contour(x, y.ravel(), sigma_pos_sq*((x - mu_neg[0])*2 + (y - mu_neg[1])2)-sigma_neg_sq((x - mu_pos[0])*2 + (y - mu_pos[1])*2)+c,[0])
plt.savefig('quadratic2.png')
plt.figure(4)
plt.scatter(x=X_pos[:,0], y=X_pos[:,1], s=10, color='red', marker='*',label='Positive example') # plot the data points from X_pos
plt.scatter(x=X_neg[:,0], y=X_neg[:,1], s=10, color='blue', marker='*',label='Negative exmaple') # plot the data points from X_neg
plt.axis([-15,42,-35,45])
plt.ylabel('Feature 2')
plt.xlabel('Feature 1')
plt.legend(loc='upper right')
