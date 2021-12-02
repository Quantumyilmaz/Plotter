from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib.table as tbl
import numpy as np
from matplotlib.lines import Line2D
from tqdm import tqdm
import matplotlib.dates as mdates

rcParams["font.family"] = 'Baskerville'
DPI = 200
#plot_indexten buraya axis gelsin sirayla for loopta yap

def plotter(arg_list, attribute_list,**kwargs):

    plot_count = len(arg_list)
    ncols= kwargs.get('ncols',2)
    nrows = int(np.ceil(plot_count/ncols))
    nlastrow = plot_count%ncols
    dpi= kwargs.get('dpi') if kwargs.get('dpi') else DPI

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,figsize=kwargs.get("figsize",(20+kwargs.get('xpad',0), 6*nrows+kwargs.get('ypad',0)))
                                                    ,gridspec_kw={'height_ratios':[1]*nrows,'width_ratios': [1]*ncols})
    fig.dpi = dpi
    if nlastrow and nrows>1:
        width = 1/(nlastrow+1)
        space = 0.5*(1/nlastrow - width)
        for i in ax[-1]:
            fig.delaxes(i)
        fig.add_axes([space,0,width,0.9/nrows])
        
            
    def get_axes(figure_axes=fig.axes,attribute_list=attribute_list,arg_list=arg_list,pipeline=1):
    
        for i, axis in tqdm(enumerate(figure_axes),total=len(figure_axes)):
            
            if pipeline:
                if kwargs.get('second_plot',0):
                    row_no = i//3+1
                    col_no = i%3+1
                    ax2=fig.add_subplot(nrows,ncols,i+1, label=f"second_{row_no}{col_no}", frame_on=False)
                    ax2.xaxis.tick_top()
                    ax2.yaxis.tick_right()
                    ax2.xaxis.set_label_position('top') 
                    ax2.yaxis.set_label_position('right')
                    ax2.mother = axis
                else:
                    axis.spines["top"].set_visible(False)
                    axis.spines["right"].set_visible(False)

                if kwargs.get('centered_y_axis'):
                    ax.spines['left'].set_position('center')
                if kwargs.get('centered_x_axis'):
                    ax.spines['bottom'].set_position('center')
            
            lines = []

            for attribute,arg in zip(attribute_list, arg_list[i]):

                if attribute[:4]=='2nd_':
                    ax_helper(ax2,attribute[4:],arg,lines)
                elif attribute=='inset_axes':
                    axins = axis.inset_axes(arg['bounds'])
                    get_axes([axins],arg['attributes'],arg['args'],arg.get('pipeline',1))
                else:
                    ax_helper(axis,attribute,arg,lines)

    get_axes()  
    fig_title = kwargs.get('fig_title','')
    fig.suptitle(fig_title, x=kwargs.get('suptitle_x',0.5),y=kwargs.get('suptitle_y',0.95),fontsize=kwargs.get('titlesize',25))
    
    if kwargs.get('hspace',0):
        plt.subplots_adjust(hspace=kwargs.get('hspace'))
    
    save_path = kwargs.get('save_path')
    if save_path and len(save_path)>3:
        save_filename = '_'.join([i[i.index(' ')+1:] for i in fig_title.split('\n')])+'.jpg'
        if save_path[-4]!='.':
            if save_path[-1]!='/':
                save_path += '/' + save_filename
            else:
                save_path += save_filename
        elif save_path[-4]=='.':
            pass
        else:
            print('Could not save file.')

        fig.savefig(save_path, dpi=dpi,bbox_inches='tight')
        # print('File saved to ' + save_path)
    
    if kwargs.get('show',0):
        plt.show()
    
    return fig

def make_labels(ax,x_or_y,prop):
    #labels = [round(i,10) for i in getattr(ax, f'get_{x_or_y}ticks')()]
    labels = getattr(ax, f'get_{x_or_y}ticks')().tolist()
    if isinstance(prop,dict):
        for key in prop:
            if key == "add":
                labels += prop[key]
                labels.sort()
                getattr(ax, f'set_{x_or_y}ticks')(labels)
            else:
                raise NotImplementedError
    else:
        if prop == 'positive':
            getattr(ax, f'set_{x_or_y}ticklabels')([*map(abs,labels)])
        else:
            raise NotImplementedError
        
def ax_helper(ax,attribute,arg,lines=[]):
    if arg == None:
        pass

    elif attribute=='make_table':
        dictionary = arg[0]
        dictionary.update({'ax':ax})
        table = tbl.table(**dictionary)
        table.auto_set_font_size(False)
        if len(arg)>1:
            KEYS = list(arg[1].keys())
            if KEYS.count('text_props'):
                table_props = table.properties()
                table_cells = table_props['children']
                for cell in table_cells:
                    #cell.PAD = 0.01
                    for prop in arg[1]['text_props']:
                        getattr(cell.get_text(), prop)(*arg[1]['text_props'][prop])
            if KEYS.count('fontsize'):
                table.set_fontsize(arg[1]['fontsize'])
            if KEYS.count('row_scale') and KEYS.count('col_scale'):
                table.scale(arg[1]['col_scale'], arg[1]['row_scale'])
            if KEYS.count("alpha"):
                for cell in table._cells:
                    table._cells[cell].set_alpha(.5)
        ax.add_table(table)
        
    elif attribute=='make_twinx':
        assert isinstance(arg,dict), "Please make sure your list contains only a dictionary with attributes as keys and arguments as values, which must be lists or dictionaries."
        twin_ax = ax.twinx()
        for attr in arg.keys():
            ax_helper(twin_ax,attr,arg[attr])
        twin_ax.spines["top"].set_visible(False)
        
    elif attribute=='make_twiny':
        twin_ax = ax.twiny()

        twin_ax.set_xlabel(**arg['xlabel'])
        twin_ax.set_xlim(ax.get_xlim())
        twin_ax.set_xticks(arg['tick_locations'])
        if list(arg.keys()).count('tick_function'):
            twin_ax.set_xticklabels(arg['tick_function'](arg['tick_locations']))
        if list(arg.keys()).count('tick_params'):
            twin_ax.tick_params(**arg['tick_params'])

    elif attribute=='ticks':
        for i in arg.keys():
            make_labels(ax,i,arg[i])
    
    elif attribute=='color_ax':
        if ax.get_label().count('second_'):
            ax.mother.spines['top'].set_color(arg['color'])
            ax.mother.spines['right'].set_color(arg['color'])
        else:
            ax.spines['bottom'].set_color(arg['color'])
            ax.spines['left'].set_color(arg['color'])
                                              
    elif attribute=='text':
        ax.text(*arg[0],**arg[1],transform=ax.transAxes)
    
    elif attribute=='set_alpha':
        ax.patch.set_alpha(arg[0])

    elif attribute.count("time_format"):
        assert len(arg) == 1
        formatter = arg[0]
        x_or_y = attribute.split('time_format')[-1]
        getattr(ax,f"{x_or_y}axis").set_major_formatter(mdates.DateFormatter(formatter))

    elif sum([isinstance(i,dict) for i in arg]):
        if attribute =='plot' and list(arg[-1].keys()).count('fillstyle'):
            line_main, = getattr(ax, attribute)(*arg[:-1],**arg[-1])
            line = Line2D(*line_main.get_data(),linestyle=arg[-1]['ls'],color=arg[-1]['color'],fillstyle=arg[-1]['fillstyle'],marker=arg[-1]['marker'], markersize=20)
            lines.append(line)
        elif attribute =='legend' and list(arg[-1].keys()).count('line_order'):
            line_order = arg[-1].pop('line_order')
            line_order = [*map(lambda x: tuple([*map(lambda i: lines[i],x)]),line_order)]
            getattr(ax, attribute)(handles = line_order,**arg[-1])
        else:
            getattr(ax, attribute)(*arg[:-1],**arg[-1])
    else:
        getattr(ax, attribute)(*arg)

        
        
        