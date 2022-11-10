package nill.morena.services;

import scala.util.parsing.combinator.testing.Str;

import java.util.Arrays;

public class MathHelper {
    public static long calls=0;
    private static long call=0;
    public static long sum=0;
    public static long max=0;
    public static long min=1000;

    public static void avgRespTime(long sec){
        calls++;
        call++;
        sum = sum + sec;
        if (sec > max ) max = sec;
        if (sec < min ) min = sec;
        if ((call % 100)==0) {
            System.out.println("Average resp. time  for last 100 calls is " + (sum / call) + "(ms) , Max :" + max + ",Min :" + min+",total call: "+calls);
            call = 0;
            sum = 0;
        }
    }

    public static String squeeze(String str){
        String[] strs = str.split(" ");
        String[] queryStr = new String[65];
        int leng = strs.length;
        int count = 0;
        int step = 3;
        for(int i=0 ; i < leng+1;){
            queryStr[count] = strs[i];
            ++count;
            i +=step;
//            if ( (count * step) > leng)  queryStr[count-1] = strs[leng - 1];
        }

        return String.join(" ",queryStr);
    }

}

