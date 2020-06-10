#include<stdio.h>



void main()
{
	int n,sum,i,j,k,temp,s=0,count=0,flag=0;
	scanf("%d %d\n",&sum,&n);
	int a[n],b[n];
	for(i=0;i<n;i++)
	{
		scanf("%d ",%a[i]);
	}
	for(i=0;i<n;i++)
		b[i]=a[i];
		
	for(i=0;i<n;i++)
	{
		for(j=i+1;j<n;j++)
		{
			if(b[i]>b[j])
			{
				temp=b[i];
				b[i]=b[j];
				b[j]=temp;
			}
		}
	}
	for(i=0;i<n;i++)
	{
		
		s=s+b[i];
		if(s<=sum)
			count++;
		else
			break;
	}
	int arr[count];
	for(i=0;i<count;i++)
	{
		for(j=0;j<n;j++)
		{
			if(b[i]==a[j])
			{
				for(k=0;k<i;k++)
					if(arr[i]==j)
						flag=1;
				
				if(flag==1)
				{
					flag=0;
					continue;
				}
				else
					arr[i]==j;
			}
		}
	}
	
	for(i=0;i<count;i++)
	{
		for(j=i+1;j<count;j++)
		{
			if(arr[i]>arr[j])
			{
				temp=arr[i];
				arr[i]=arr[j];
				arr[j]=temp;
			}
		}
	}
	printf(%d\n",count);
	for(i=0;i<count;i++)
		printf("%d ",arr[i]);
		
}