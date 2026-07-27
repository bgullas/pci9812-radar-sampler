// PCI_9812 : Continuous AI  (syntax-fixed)
#include <stdio.h>
#include <windows.h>
#include "dask.h"

// ?? Constants ????????????????????????????????????????????????????????????????
#define CardNumber   0
#define LastADChan   1               // CH0�CH3 (4 channels)
#define ADChanCount  2
#define ScanCount    65536           // scans per half-buffer

// At 20 MS/s: half = 65536 / 20 000 000 ? 3.3 ms
// At 20 kS/s: half = 65536 / 20 000    ? 3.3 s  (lower rate needs smaller ScanCount)
#define SampleRate   1000000.0       // S/s per channel (4ch = 8 MSPS total)

#define ClkSel       (P9812_AD2_GT_PCI | P9812_CLKSRC_INT)
#define TriggerLevel 0x7F             // 0x7F
#define PostCount    0
#define RunSeconds   2

// ?? Variables ????????????????????????????????????????????????????????????????
I16     cardID = -1;
I16     err = 0;
BOOLEAN HalfReady = 0;
BOOLEAN fstop = 0;              // FIX: was missing � used in inner loop
U32     AccessCnt = 0;
U32     MemSize = 0;
char* file_name = "P9812d_Jul14";       // driver appends .dat

static volatile BOOL g_stop = FALSE;

BOOL WINAPI CtrlHandler(DWORD dwCtrlType) {
    g_stop = TRUE;
    return TRUE;   // suppress default termination so cleanup runs
}

int main(void)
{
    SetConsoleCtrlHandler(CtrlHandler, TRUE);

    // 1. Register card
    cardID = Register_Card(PCI_9812, CardNumber);
    if (cardID < 0) {
        printf("Register_Card error=%d\n", cardID);
        return 1;
    }

    // 2. Query available DMA memory
    err = AI_InitialMemoryAllocated(cardID, &MemSize);
    if (err != NoError) {
        printf("AI_InitialMemoryAllocated error=%d\n", err);
        Release_Card(cardID);
        return 1;
    }
    printf("DMA memory available: %u KB\n", MemSize);
    if (MemSize * 1024 < (U32)(ScanCount * ADChanCount * sizeof(I16))) {
        printf("WARNING: available DMA memory (%u KB) is smaller than "
            "requested buffer (%u B). Reduce ScanCount or increase the "
            "DMA buffer size in ADLINK Device Manager.\n",
            MemSize, ScanCount * ADChanCount * (U32)sizeof(I16));
        // Continue anyway � driver may still work with a smaller effective buffer
    }

    // 3. Configure the ADC (must be before AI_AsyncDblBufferMode)
    err = AI_9812_Config(cardID,
        P9812_TRGMOD_SOFT,
        P9812_TRGSRC_CH0,
        P9812_TRGSLP_POS,
        ClkSel,
        TriggerLevel,
        0);
    if (err != NoError) {
        printf("AI_9812_Config error=%d\n", err);
        Release_Card(cardID);
        return 1;
    }

    // 4. Enable double-buffer mode (must be after AI_9812_Config)
    err = AI_AsyncDblBufferMode(cardID, 1);
    if (err != NoError) {
        printf("AI_AsyncDblBufferMode error=%d\n", err);
        Release_Card(cardID);
        return 1;
    }

    // 5. Start continuous scan to file
    err = AI_ContScanChannelsToFile(cardID,
        LastADChan,
        AD_B_1_V,
        file_name,
        (U32)(ScanCount * ADChanCount),
        SampleRate,
        ASYNCH_OP);
    if (err != NoError) {
        printf("AI_ContScanChannelsToFile error=%d\n", err);
        Release_Card(cardID);
        return 1;
    }

    printf("Acquiring to %s.dat for %d seconds...\n", file_name, RunSeconds);

    DWORD t_start = GetTickCount();
    U32   transfers = 0;

    // 6. Poll loop: wait for each half, flush it to file
    do {
        HalfReady = 0;
        do {
            AI_AsyncDblBufferHalfReady(cardID, &HalfReady, &fstop);
        } while (!HalfReady && !g_stop &&
                 (GetTickCount() - t_start) < (DWORD)(RunSeconds * 1000));

        if (!HalfReady) {
            printf("\n[DBG] inner loop exited without HalfReady "
                   "(elapsed=%lums, transfers=%u)\n",
                   (unsigned long)(GetTickCount() - t_start), transfers);
            break;
        }

        AI_AsyncDblBufferTransfer(cardID, NULL);   // flush half ? .dat
        transfers++;
        printf("\r[DBG] transfers=%u  elapsed=%lus",
               transfers, (unsigned long)(GetTickCount() - t_start) / 1000);
        fflush(stdout);

    } while (!fstop && !g_stop &&
             (GetTickCount() - t_start) < (DWORD)(RunSeconds * 1000));

    // 7. Stop acquisition
    err = AI_AsyncClear(cardID, &AccessCnt);
    if (err != NoError) {
        printf("\nAI_AsyncClear error=%d\n", err);
    }
    printf("\nDone. %u samples transferred to %s.dat\n", AccessCnt, file_name);

    Release_Card(cardID);
    return 0;
}
