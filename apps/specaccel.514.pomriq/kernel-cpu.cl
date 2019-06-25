#define PIx2 6.2831853071795864769252867665590058f

struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};

__kernel void ComputePhiMag(int numK, __global float* restrict phiR, __global float* restrict phiI, __global float* restrict phiMag) {
    int indexK = get_global_id(0);
    float real = phiR[indexK];
    float imag = phiI[indexK];
    phiMag[indexK] = real*real + imag*imag;
}

__kernel void ComputeQ(int numK, int numX, __global struct kValues* restrict kVals, __global float* restrict x, __global float* restrict y, __global float* restrict z, __global float* restrict Qr, __global float* restrict Qi) {
    int indexX = get_global_id(0);

    float QrSum = 0.0;
    float QiSum = 0.0;

    for (int indexK = 0; indexK < numK; indexK++) {
        float expArg = PIx2 * (kVals[indexK].Kx * x[indexX] +
                kVals[indexK].Ky * y[indexX] +
                kVals[indexK].Kz * z[indexX]);

        float cosArg = cos(expArg);
        float sinArg = sin(expArg);

        float phi = kVals[indexK].PhiMag;
        QrSum += phi * cosArg;
        QiSum += phi * sinArg;
    }

    Qr[indexX] += QrSum;
    Qi[indexX] += QiSum;
}
