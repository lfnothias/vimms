using System;
using IAPI_Assembly;
using Thermo.Interfaces.FusionAccess_V1;
using Thermo.Interfaces.FusionAccess_V1.MsScanContainer;
using Thermo.Interfaces.InstrumentAccess_V1.MsScanContainer;
using Thermo.Interfaces.SpectrumFormat_V1;

namespace IAPI_Console
{
    class Program
    {
        static void Main(string[] args)
        {
            // Use the Factory creation method to create a Fusion Access Container
            // IFusionInstrumentAccessContainer fusionContainer = Factory<IFusionInstrumentAccessContainer>.Create();

            // Above won't work without a license! For testing, use the following FusionContainer that loads data from an mzML file.
            string filename = "C:\\Users\\joewa\\Work\\git\\clms\\test_data\\Beer_1_full1.mzML";
            IFusionInstrumentAccessContainer fusionContainer = new FusionContainer(filename);

            // Connect to the service by going 'online'
            fusionContainer.StartOnlineAccess();

            // Wait until the service is connected 
            // (better through the event, but this is nice and simple)
            while (!fusionContainer.ServiceConnected)
            {
                ;
            }

            // From the instrument container, get access to a particular instrument
            IFusionInstrumentAccess fusionAccess = fusionContainer.Get(1);

            // Get the MS Scan Container from the fusion
            IFusionMsScanContainer fusionScanContainer = fusionAccess.GetMsScanContainer(0);

            // Register to MsScanArrived event
            fusionScanContainer.MsScanArrived += FusionScanContainer_MsScanArrived;
            Console.ReadLine();

            // TODO: add more stuff from file:///C:/Users/joewa/Work/thermo_docs/IAPI_tribrid_docs/lizlg-u48yn_files/25a513ae-e1d1-c8e6-d93a-978cbb684ec3.htm
        }

        private static void FusionScanContainer_MsScanArrived(object sender, MsScanEventArgs e)
        {
            Thermo.Interfaces.InstrumentAccess_V1.MsScanContainer.IMsScan scan = e.GetScan();
            Console.WriteLine("[{0:HH:mm:ss.ffff}] Received MS Scan Number {1} -- {2} peaks",
                DateTime.Now,
                scan.Header["Scan"],
                scan.CentroidCount);
            foreach (ICentroid centroid in scan.Centroids)
            {
                Console.WriteLine("{0} {1}", centroid.Mz, centroid.Intensity);
            }
        }

    }
}
