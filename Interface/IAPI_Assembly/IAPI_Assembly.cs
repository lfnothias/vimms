using PSI_Interface.MSData;
using System;
using System.Linq;
using Thermo.Interfaces.FusionAccess_V1;
using Thermo.Interfaces.FusionAccess_V1.MsScanContainer;
using Thermo.Interfaces.InstrumentAccess_V1.MsScanContainer;
using Thermo.Interfaces.FusionAccess_V1.Control;
using Thermo.Interfaces.InstrumentAccess_V1;
using Thermo.Interfaces.InstrumentAccess_V1.AnalogTraceContainer;
using Thermo.Interfaces.InstrumentAccess_V1.Control;
using System.Collections.Generic;
using static PSI_Interface.MSData.SimpleMzMLReader;
using Thermo.Interfaces.SpectrumFormat_V1;
using System.Threading.Tasks;

namespace IAPI_Assembly
{
    public class FusionContainer : IFusionInstrumentAccessContainer
    {

        private readonly string filename;
        private readonly IEnumerable<SimpleSpectrum> _spectra;
        public bool ServiceConnected { get; } = true;
        public event EventHandler ServiceConnectionChanged;
        public event EventHandler<MessagesArrivedEventArgs> MessagesArrived;

        public FusionContainer(string filename)
        {
            this.filename = filename;
            Console.WriteLine("filename is " + filename);
            if (filename != null)
            {
                SimpleMzMLReader reader = new SimpleMzMLReader(filename);
                Console.WriteLine("NumSpectra = " + reader.NumSpectra);
                _spectra = reader.ReadAllSpectra();
            }
        }

        public void Dispose()
        {
            throw new NotImplementedException();
        }

        public IFusionInstrumentAccess Get(int index)
        {
            return new FusionAccess(_spectra);
        }

        public void StartOnlineAccess()
        {
            Console.WriteLine("Going online!");
        }

        IInstrumentAccess IInstrumentAccessContainer.Get(int index)
        {
            throw new NotImplementedException();
        }
    }

    class FusionAccess : IFusionInstrumentAccess
    {
        private IEnumerable<SimpleSpectrum> _spectra;

        public FusionAccess(IEnumerable<SimpleSpectrum> spectra)
        {
            _spectra = spectra;
        }

        public IFusionControl Control => throw new NotImplementedException();

        public int InstrumentId => throw new NotImplementedException();

        public string InstrumentName => throw new NotImplementedException();

        public bool Connected => throw new NotImplementedException();

        public int CountMsDetectors => throw new NotImplementedException();

        public string[] DetectorClasses => throw new NotImplementedException();

        public int CountAnalogChannels => throw new NotImplementedException();

        IControl IInstrumentAccess.Control => throw new NotImplementedException();

        public event EventHandler<ContactClosureEventArgs> ContactClosureChanged;
        public event EventHandler ConnectionChanged;
        public event EventHandler<AcquisitionErrorsArrivedEventArgs> AcquisitionErrorsArrived;

        public IAnalogTraceContainer GetAnalogTraceContainer(int analogDetectorSet)
        {
            throw new NotImplementedException();
        }

        public IFusionMsScanContainer GetMsScanContainer(int msDetectorSet)
        {
            return new MsScanContainer(_spectra);
        }

        IMsScanContainer IInstrumentAccess.GetMsScanContainer(int msDetectorSet)
        {
            return new MsScanContainer(_spectra);
        }
    }

    class MsScanContainer : IFusionMsScanContainer
    {
        private SimpleSpectrum[] spectra;
        private IMsScan lastScan;

        public MsScanContainer(IEnumerable<SimpleSpectrum> spectra)
        {
            this.spectra = spectra.ToArray();
            for (int i = 0; i < this.spectra.Length; i++)
            {
                // schedule events to be triggered, see https://stackoverflow.com/questions/545533/delayed-function-calls
                SimpleSpectrum current = this.spectra[i];
                int elutionTimeInMilliSecond = (int)(current.ElutionTime * 60 * 1000);
                IMsScan msScan = new MyMsScan(current);
                MsScanEventArgs args = new MyMsScanEventArgs(msScan);
                Task.Delay(elutionTimeInMilliSecond).ContinueWith(t => OnMsScanArrived(args));
                this.lastScan = msScan;
            }
        }

        protected virtual void OnMsScanArrived(MsScanEventArgs e)
        {
            EventHandler<MsScanEventArgs> handler = MsScanArrived;
            if (handler != null)
            {
                handler(this, e);
            }
        }

        public string DetectorClass => throw new NotImplementedException();

        public event EventHandler<MsScanEventArgs> MsScanArrived;

        public IMsScan GetLastMsScan()
        {
            return this.lastScan;
        }
    }

    class MyMsScanEventArgs : MsScanEventArgs
    {
        private IMsScan msScan;

        public MyMsScanEventArgs(IMsScan msScan)
        {
            this.msScan = msScan;
        }

        public override IMsScan GetScan()
        {
            return this.msScan;
        }
    }

    class MyMsScan : IMsScan
    {
        public IDictionary<string, string> Header { get; } = new Dictionary<string, string>();

        public IInformationSourceAccess TuneData => throw new NotImplementedException();

        public IInformationSourceAccess Trailer => throw new NotImplementedException();

        public IInformationSourceAccess StatusLog => throw new NotImplementedException();

        public string DetectorName => throw new NotImplementedException();

        public int? NoiseCount => throw new NotImplementedException();

        public IEnumerable<INoiseNode> NoiseBand => throw new NotImplementedException();

        public int? CentroidCount { get; } = 0;

        public IEnumerable<ICentroid> Centroids { get; }

        public IChargeEnvelope[] ChargeEnvelopes => throw new NotImplementedException();

        #region IDisposable Support
        private bool disposedValue = false; // To detect redundant calls

        public MyMsScan(SimpleSpectrum scan)
        {
            this.Header["Scan"] = scan.ScanNumber.ToString();
            List<ICentroid> myList = new List<ICentroid>();
            if (scan.Centroided)
            {
                this.CentroidCount = scan.Mzs.Length;
                foreach (Peak p in scan.Peaks)
                {
                    ICentroid centroid = new Centroid(p);
                    myList.Add(centroid);
                }
                this.Centroids = myList;
            }
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // TODO: dispose managed state (managed objects).
                }

                // TODO: free unmanaged resources (unmanaged objects) and override a finalizer below.
                // TODO: set large fields to null.

                disposedValue = true;
            }
        }

        // TODO: override a finalizer only if Dispose(bool disposing) above has code to free unmanaged resources.
        // ~MyMsScan() {
        //   // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
        //   Dispose(false);
        // }

        // This code added to correctly implement the disposable pattern.
        public void Dispose()
        {
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            Dispose(true);
            // TODO: uncomment the following line if the finalizer is overridden above.
            // GC.SuppressFinalize(this);
        }
        #endregion

    }

    class Centroid : ICentroid
    {
        public Centroid(Peak p)
        {
            this.Mz = p.Mz;
            this.Intensity = p.Intensity;
        }

        public bool? IsExceptional => throw new NotImplementedException();

        public bool? IsReferenced => throw new NotImplementedException();

        public bool? IsMerged => throw new NotImplementedException();

        public bool? IsFragmented => throw new NotImplementedException();

        public int? Charge => throw new NotImplementedException();

        public IMassIntensity[] Profile => throw new NotImplementedException();

        public double? Resolution => throw new NotImplementedException();

        public int? ChargeEnvelopeIndex => throw new NotImplementedException();

        public bool? IsMonoisotopic => throw new NotImplementedException();

        public bool? IsClusterTop => throw new NotImplementedException();

        public double Mz { get; }

        public double Intensity { get; }
    }

}
