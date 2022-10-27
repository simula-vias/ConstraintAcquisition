package nill.morena.services;

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

}

