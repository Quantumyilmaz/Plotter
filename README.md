# Plotter
Matpllotlib based framework to create plots

## How it works

### Base example

- Import

```
from plotter import plotter

```


- Create a list of arguments belonging to your plot:
  
```
  args =[ #For each subplot open square brackets
          [#For each matplotlib method you want to call e.g. .plot(), open square brackets and pass in the data.
          [range(4)]
          ]             
        ]
```

- Create a list of matplotlib method you want to call. Here you can call any method available for [matplotlib Axes class](https://matplotlib.org/stable/api/axes_api.html):

```
  methods =['plot']
```

- Call plotter:
    * ncols defines the number of columns in your plot, i.e. number of subplots per row.
    * ```show=1``` calls ```matplotlib.pyplot.show()``` a.k.a. ```plt.show()```
    * if ```save_path``` is given as keyword argument, the plot will be saved to the provided path.

```
plotter(args,methods,ncols=1,show=1);
```

![image info](./base_example.png)