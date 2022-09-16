import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import emoji as emoji


class Chat:
    """
    A class to represent a person.

    ...

    Attributes
    ----------
    data : pd.Dataframe
        dataframe containing the whole chat
    daily_data : pd.Dataframe
        data aggregated by Day, Speaker and Weekday
    monthly_data : pd.Dataframe
        data aggregated by Month and Speaker
    file : str
        filename of the underlying chat backup .txt

    Methods
    -------
    print():
        Prints the whole chat in a formatted way

    print_grouped_data():
        Prints the grouped data in a formatted way
    """

    def __init__(self, file, time_format='HH:MM', remove_last_names=True):
        """
        Constructs all the necessary attributes for the person object.

        Note: Remove the Lines concerning "encryption" and "number changes" in the .txt file

        Parameters
        ----------
            file : str
                location and filename of the whatsapp chat backup .txt
            time_format : str
                time format used by the chat backup. E.g. 24 hour or 12 hour AM/PM
            remove_last_names : bool
                whether to remove all Speakers' lastnames

        """

        self.file = file

        if time_format == 'HH:MM':
            def use_regex(input_text):
                pattern = re.compile(r'[0-9]+/[0-9]+/[0-9]+,\s\d\d:\d\d\s-')
                return pattern.match(input_text)
        else:
            # Job for someone else
            pass

        current_speaker = None
        current_date = None
        current_time = None
        current_message = None

        Dates = []
        Times = []
        Speakers = []
        Messages = []

        with open(file, encoding="utf-8") as fp:
            Lines = fp.readlines()
            for line in Lines:
                if use_regex(line):
                    current_date = line.split(',', 1)[0]
                    current_time = line.split(',', 1)[1].split('-', 1)[0][1:-1]
                    current_speaker = line.split(',', 1)[1].split('-', 1)[1].split(':', 1)[0][1:]
                    current_message = line.split(',', 1)[1].split('-', 1)[1].split(':', 1)[1][1:-1]

                else:
                    current_message = line

                Dates.append(current_date)
                Times.append(current_time)
                Speakers.append(current_speaker)
                Messages.append(current_message)

        self.data = pd.DataFrame({'Date': Dates, 'Time': Times, 'Speaker': Speakers, 'Message': Messages})
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data['Time'] = pd.to_datetime(self.data['Time'], format='%H:%M')  # .dt.time
        self.data.insert(2, 'Weekday', self.data['Date'].dt.dayofweek)
        self.data['Weekday'].replace([0, 1, 2, 3, 4, 5, 6],
                                     ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                                     inplace=True)

        self.data['Message Length'] = self.data['Message'].str.len()

        if remove_last_names:
            self.data['Speaker'] = self.data['Speaker'].str.split(' ').str[0]

        # Kann man eigentlich weglassen
        self.data['Emoji Count'] = [emoji.emoji_count(message) for message in self.data.Message]
        self.data['Contains Emoji'] = self.data['Emoji Count'].where(self.data['Emoji Count'] == 0, 1)
        self.data['Emoji Density'] = self.data['Emoji Count'] / self.data['Message Length']

        # Chat data aggregated over every day
        self.daily_data = self.data.groupby(['Date', 'Speaker', 'Weekday']).agg(
            {'Message': ['count'], 'Message Length': ['sum'], 'Contains Emoji': ['mean']})
        self.daily_data.columns = self.daily_data.columns.get_level_values(
            0) + '_' + self.daily_data.columns.get_level_values(1)
        self.daily_data.reset_index(inplace=True)

        # Chat data aggregated over every month
        self.monthly_data = self.data.groupby([pd.Grouper(key='Date', freq='M'), 'Speaker']).agg(
            {'Message Length': ['mean'], 'Contains Emoji': ['mean'], 'Emoji Density': ['mean']})
        self.monthly_data.columns = self.monthly_data.columns.get_level_values(
            0) + '_' + self.monthly_data.columns.get_level_values(1)
        self.monthly_data.reset_index(inplace=True)

    def print(self):
        """
        Prints the whole chat in a formatted way

        Parameters
        ----------

        Returns
        -------
        None
        """

        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):  # more options can be specified also
            print(self.data)

    def print_grouped_data(self):
        """
        Prints the whole chat in a formatted way

        Parameters
        ----------

        Returns
        -------
        None
        """

        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):  # more options can be specified also
            print(self.daily_data)

    def highest_activity(self):
        """
        Returns a pandas dataframe with the highest chat activity data for every speaker

        Parameters
        ----------

        Returns
        -------
        Pandas Dataframe
        """
        unique_highest = []

        for speaker in self.daily_data['Speaker'].unique():
            unique_highest.append(self.daily_data.loc[self.daily_data.loc[self.daily_data['Speaker'] == speaker][
                'Message Length_sum'].idxmax()])

        return pd.concat(unique_highest, axis=1).transpose().reset_index(drop=True)

    def plot_monthly_messages(self, fig_size=(7.1, 4), save=None, title='Monthly Messages Histogram and KDE',
                              xlim=None, speaker_order=None):
        """
        Plots and saves a Seaborn histplot of the chat data grouped into monthly bins
        depicting the absolute number of messages in that month

        Parameters
        ----------
        fig_size : tuple
            tuple depicting the size of the resulting matplotlib figure
            default is (7.1, 4) and optimized for PowerPoint usage
        save : str
            location and file extension to pass to the plt.savefig() function
        title : str
            title of the resulting plot
        xlim : tuple
            X-Axis Datetime Limits for the Plot
        speaker_order : List
            Order in which the Participants are to be depicted

        Returns
        -------
        Pandas Dataframe
        """
        sns.set()
        plt.figure(figsize=fig_size)

        if speaker_order is not None and isinstance(speaker_order, list):
            ax = sns.histplot(data=self.data[['Date', 'Speaker', 'Message']], x='Date', hue='Speaker', binwidth=31,
                              kde=True, hue_order=speaker_order)
        else:
            ax = sns.histplot(data=self.data[['Date', 'Speaker', 'Message']], x='Date', hue='Speaker', binwidth=31,
                              kde=True, hue_order=sorted(self.data.Speaker.unique()))

        if xlim is not None and len(xlim) == 2:
            plt.xlim(pd.to_datetime(xlim[0]), pd.to_datetime(xlim[1]))

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
        date_form = DateFormatter("%Y")
        ax.xaxis.set_major_formatter(date_form)
        plt.xlabel('Date')
        plt.ylabel('Monthly Messages')
        plt.title(title)
        plt.tight_layout()

        if save is None or not isinstance(save, str):
            plt.savefig('Monthly Messages Histogram and KDE.png', dpi=600)
        else:
            plt.savefig(save, dpi=600)
        plt.show()

    def plot_hourly_messages(self, fig_size=(7.1, 4), save=None, title='Hourly Distribution of Messages and KDE',
                             speaker_order=None):
        """
        Plots and saves a Seaborn histplot of the chat data grouped into hourly bins
        depicting the relative hourly messages normalized for each participant.

        Parameters
        ----------
        fig_size : tuple
            tuple depicting the size of the resulting matplotlib figure
            default is (7.1, 4) and optimized for PowerPoint usage
        save : str
            location and file extension to pass to the plt.savefig() function
        title : str
            title of the resulting plot
        speaker_order : List
            Order in which the Participants are to be depicted

        Returns
        -------
        Pandas Dataframe
        """

        plt.figure(figsize=fig_size)
        if speaker_order is not None and isinstance(speaker_order, list):
            ax = sns.histplot(data=self.data[['Time', 'Speaker', 'Message']],
                              hue_order=speaker_order, x='Time', hue='Speaker', bins=48, kde=True,
                              element='step', stat='percent', common_norm=False)
        else:
            ax = sns.histplot(data=self.data[['Time', 'Speaker', 'Message']],
                              hue_order=sorted(self.data.Speaker.unique()), x='Time', hue='Speaker', bins=48, kde=True,
                              element='step', stat='percent', common_norm=False)

        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        date_form = DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(date_form)
        plt.xlabel('Hour')
        plt.ylabel('Percent of Messages')
        plt.title(title)
        legend = ax.get_legend()
        handles = legend.legendHandles
        legend.remove()
        ax.legend(handles, sorted(self.data.Speaker.unique()), title='Speaker', loc='upper left')
        plt.tight_layout()

        if save is None or not isinstance(save, str):
            plt.savefig('Hourly Distribution of Messages and KDE.png', dpi=600)
        else:
            plt.savefig(save, dpi=600)
        plt.show()

    def plot_absolute_history(self, fig_size=(7.1, 4), save=None, title='History of Message Intensity', xlim=None,
                              speaker_order=None):
        """
        Plots and saves a Seaborn lineplot depiciting
        the message intensity over the complete chat history for each participant.

        Parameters
        ----------
        fig_size : tuple
            tuple depicting the size of the resulting matplotlib figure
            default is (7.1, 4) and optimized for PowerPoint usage
        save : str
            location and file extension to pass to the plt.savefig() function
        title : str
            title of the resulting plot
        xlim : tuple
            X-Axis Datetime Limits for the Plot
        speaker_order : List
            Order in which the Participants are to be depicted

        Returns
        -------
        Pandas Dataframe
        """
        sns.set()
        plt.figure(figsize=fig_size)
        if speaker_order is not None and isinstance(speaker_order, list):
            g = sns.FacetGrid(self.daily_data, hue='Speaker', col='Speaker', height=4, aspect=7.1 / 8,
                              legend_out=False, hue_order=speaker_order, col_wrap=2, col_order=speaker_order)
            g.map(sns.lineplot, 'Date', 'Message Length_sum')
        else:
            g = sns.FacetGrid(self.daily_data, hue='Speaker', col='Speaker', height=4, aspect=7.1 / 8,
                              legend_out=False, hue_order=sorted(self.data.Speaker.unique()), col_wrap=2,
                              col_order=sorted(self.data.Speaker.unique()))
            g.map(sns.lineplot, 'Date', 'Message Length_sum')

        if xlim is not None and len(xlim) == 2:
            plt.xlim(pd.to_datetime(xlim[0]), pd.to_datetime(xlim[1]))

        g.add_legend(loc='upper left')
        g.set_ylabels('Cumulated Daily Characters')
        plt.suptitle(title)
        plt.tight_layout()

        if save is None or not isinstance(save, str):
            plt.savefig('History of Message Intensity.png', dpi=600)
        else:
            plt.savefig(save, dpi=600)
        plt.show()

    def plot_daily_emoji_usage(self, fig_size=(7.1, 4), save=None, title='Daily Emoji Usage', speaker_order=None):
        """
        Plots and saves a Seaborn barplot depicting
        the emoji usage for every weekday normalized for every participant.

        Parameters
        ----------
        fig_size : tuple
            tuple depicting the size of the resulting matplotlib figure
            default is (7.1, 4) and optimized for PowerPoint usage
        save : str
            location and file extension to pass to the plt.savefig() function
        title : str
            title of the resulting plot
        speaker_order : List
            Order in which the Participants are to be depicted

        Returns
        -------
        Pandas Dataframe
        """
        plt.figure(figsize=fig_size)

        if speaker_order is not None and isinstance(speaker_order, list):
            sns.barplot(self.data, x='Weekday', y='Contains Emoji', hue='Speaker',
                        hue_order=speaker_order,
                        order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        else:
            sns.barplot(self.data, x='Weekday', y='Contains Emoji', hue='Speaker',
                        hue_order=sorted(self.data.Speaker.unique()),
                        order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

        plt.ylabel('Messages containing Emojis')
        plt.legend(loc='upper left')
        plt.title(title)
        plt.tight_layout()
        if save is None or not isinstance(save, str):
            plt.savefig('Daily Emoji Usage.png', dpi=600)
        else:
            plt.savefig(save, dpi=600)
        plt.show()

    def plot_daily_messages(self, fig_size=(7.1, 4), save=None, title='Daily Messages', speaker_order=None):
        """
        Plots and saves a Seaborn barplot depicting
        the amount of messages for every weekday for every participant.

        Parameters
        ----------
        fig_size : tuple
            tuple depicting the size of the resulting matplotlib figure
            default is (7.1, 4) and optimized for PowerPoint usage
        save : str
            location and file extension to pass to the plt.savefig() function
        title : str
            title of the resulting plot
        speaker_order : List
            Order in which the Participants are to be depicted

        Returns
        -------
        Pandas Dataframe
        """
        plt.figure(figsize=fig_size)
        if speaker_order is not None and isinstance(speaker_order, list):
            sns.barplot(self.daily_data, x='Weekday', y='Message_count', hue='Speaker',
                        hue_order=speaker_order,
                        order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        else:
            sns.barplot(self.daily_data, x='Weekday', y='Message_count', hue='Speaker',
                        hue_order=sorted(self.data.Speaker.unique()),
                        order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

        plt.ylabel('Messages')
        plt.legend(loc='upper left')
        plt.title(title)
        plt.tight_layout()
        if save is None or not isinstance(save, str):
            plt.savefig('Daily Messages.png', dpi=600)
        else:
            plt.savefig(save, dpi=600)
        plt.show()

    def plot_daily_messages_lengths(self, fig_size=(7.1, 4), save=None, title='Daily Message Lengths',
                                    speaker_order=None):
        """
        Plots and saves a Seaborn barplot depicting
        the length of messages for every weekday for every participant.

        Parameters
        ----------
        fig_size : tuple
            tuple depicting the size of the resulting matplotlib figure
            default is (7.1, 4) and optimized for PowerPoint usage
        save : str
            location and file extension to pass to the plt.savefig() function
        title : str
            title of the resulting plot
        speaker_order : List
            Order in which the Participants are to be depicted

        Returns
        -------
        Pandas Dataframe
        """
        plt.figure(figsize=fig_size)
        if speaker_order is not None and isinstance(speaker_order, list):
            sns.barplot(self.data, x='Weekday', y='Message Length', hue='Speaker',
                        hue_order=speaker_order,
                        order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        else:
            sns.barplot(self.data, x='Weekday', y='Message Length', hue='Speaker',
                        hue_order=sorted(self.data.Speaker.unique()),
                        order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

        plt.ylabel('Characters')
        plt.legend(loc='upper left')
        plt.title(title)
        plt.tight_layout()
        if save is None or not isinstance(save, str):
            plt.savefig('Daily Message Lengths.png', dpi=600)
        else:
            plt.savefig(save, dpi=600)
        plt.show()
